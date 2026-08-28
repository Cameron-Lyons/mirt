from __future__ import annotations

from dataclasses import dataclass, field
from math import lgamma
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import (
    PROB_CLIP_MAX,
    PROB_CLIP_MIN,
    PROB_EPSILON,
    REGULARIZATION_EPSILON,
)
from mirt.estimation.base import BaseEstimator
from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def _finite_float(value: object, name: str) -> float:
    """Validate a finite real-valued control."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise MirtValidationError(
            f"{name} must be a finite real number",
            parameter=name,
            value=value,
        )
    numeric = float(value)
    if not np.isfinite(numeric):
        raise MirtValidationError(
            f"{name} must be a finite real number",
            parameter=name,
            value=value,
        )
    return numeric


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
        threshold = _finite_float(threshold, "threshold")
        if threshold < 0.0:
            raise MirtValidationError(
                "threshold must be non-negative",
                parameter="threshold",
                value=threshold,
                expected=">= 0",
            )
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
        lambda_0 = _finite_float(lambda_0, "lambda_0")
        lambda_1 = _finite_float(lambda_1, "lambda_1")
        theta = _finite_float(theta, "theta")
        if lambda_0 <= 0 or lambda_1 <= 0:
            raise MirtValidationError("lambda_0 and lambda_1 must be positive")
        if lambda_0 >= lambda_1:
            raise MirtValidationError("lambda_0 must be smaller than lambda_1")
        if not 0 < theta < 1:
            raise MirtValidationError("theta must be between 0 and 1")
        if not isinstance(adaptive, (bool, np.bool_)):
            raise MirtValidationError(
                "adaptive must be a boolean",
                parameter="adaptive",
                value=adaptive,
            )

        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.theta = theta
        self.adaptive = bool(adaptive)

    def log_pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute log marginal density (mixture of two Laplace)."""
        abs_x = np.abs(x)

        log_spike = (
            np.log(1 - self.theta) - np.log(2 * self.lambda_0) - abs_x / self.lambda_0
        )
        log_slab = (
            np.log(self.theta) - np.log(2 * self.lambda_1) - abs_x / self.lambda_1
        )

        return np.logaddexp(log_spike, log_slab)

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

        log_spike = (
            np.log1p(-self.theta) - np.log(2.0 * self.lambda_0) - abs_x / self.lambda_0
        )
        log_slab = (
            np.log(self.theta) - np.log(2.0 * self.lambda_1) - abs_x / self.lambda_1
        )
        gamma = np.exp(log_slab - np.logaddexp(log_spike, log_slab))

        return np.clip(gamma, 0.0, 1.0)

    def compute_effective_penalty(
        self,
        gamma: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute effective LASSO penalty given inclusion probabilities.

        For SSL, the effective penalty interpolates between spike and slab:
        lambda_eff = (1-gamma) * (1/lambda_0) + gamma * (1/lambda_1)

        The returned rate is scaled by the local quadratic curvature before
        soft-thresholding.
        """
        return (1 - gamma) / self.lambda_0 + gamma / self.lambda_1

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
        gamma = np.asarray(gamma, dtype=np.float64)
        if gamma.size == 0 or not np.all(np.isfinite(gamma)):
            raise MirtValidationError("gamma must contain finite values")
        if np.any((gamma < 0.0) | (gamma > 1.0)):
            raise MirtValidationError("gamma values must be between 0 and 1")
        if self.adaptive:
            self.theta = float(np.clip(np.mean(gamma), PROB_CLIP_MIN, PROB_CLIP_MAX))

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

        if (
            isinstance(k_max, (bool, np.bool_))
            or not isinstance(k_max, Integral)
            or k_max < 1
        ):
            raise MirtValidationError(
                "k_max must be at least 1 and an integer",
                parameter="k_max",
                value=k_max,
                expected="positive integer",
            )
        if (
            isinstance(n_inner_iter, (bool, np.bool_))
            or not isinstance(n_inner_iter, Integral)
            or n_inner_iter < 1
        ):
            raise MirtValidationError(
                "n_inner_iter must be a positive integer",
                parameter="n_inner_iter",
                value=n_inner_iter,
                expected="positive integer",
            )
        sparsity_threshold = _finite_float(
            sparsity_threshold,
            "sparsity_threshold",
        )
        if not 0.0 <= sparsity_threshold <= 1.0:
            raise MirtValidationError(
                "sparsity_threshold must be between 0 and 1",
                parameter="sparsity_threshold",
                value=sparsity_threshold,
                expected="[0, 1]",
            )
        if not isinstance(adaptive_theta, (bool, np.bool_)):
            raise MirtValidationError(
                "adaptive_theta must be a boolean",
                parameter="adaptive_theta",
                value=adaptive_theta,
            )

        self.k_max: int = int(k_max)
        self.sparsity_threshold: float = sparsity_threshold
        self.n_inner_iter: int = int(n_inner_iter)

        self._ssl_prior: SpikeSlabLassoPrior = SpikeSlabLassoPrior(
            lambda_0=lambda_0,
            lambda_1=lambda_1,
            theta=theta,
            adaptive=bool(adaptive_theta),
        )

        self._loadings: NDArray[np.float64] | None = None
        self._intercepts: NDArray[np.float64] | None = None
        self._gamma: NDArray[np.float64] | None = None
        self._mu: NDArray[np.float64] | None = None
        self._sigma: NDArray[np.float64] | None = None
        self._xi: NDArray[np.float64] | None = None
        self._elbo_history: list[float] = []
        self._fixed_loadings: bool = False

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
        if model.n_factors != self.k_max:
            raise MirtValidationError(
                "model.n_factors must match k_max",
                parameter="k_max",
                value=self.k_max,
                expected=str(model.n_factors),
            )

        responses = self._validate_responses(responses, model.n_items)
        n_persons, n_items = responses.shape

        prior_mean, prior_cov = self._validate_prior(prior_mean, prior_cov)
        prior_cov_inv = np.linalg.solve(prior_cov, np.eye(self.k_max))
        self._fixed_loadings = model.model_name == "1PL"

        self._initialize_parameters(
            responses,
            n_items,
            fixed_loadings=self._fixed_loadings,
        )

        self._mu = np.broadcast_to(prior_mean, (n_persons, self.k_max)).copy()
        self._sigma = np.broadcast_to(
            prior_cov,
            (n_persons, self.k_max, self.k_max),
        ).copy()
        self._xi = np.ones((n_persons, n_items))

        self._convergence_history = []
        self._elbo_history = []
        prev_elbo = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            self._e_step(responses, prior_mean, prior_cov_inv)

            current_elbo = self._compute_elbo(responses, prior_mean, prior_cov)
            self._elbo_history.append(current_elbo)
            self._convergence_history.append(current_elbo)

            self._log_iteration(iteration, current_elbo, elbo=current_elbo)

            if self._check_convergence(prev_elbo, current_elbo):
                converged = True
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_elbo = current_elbo

            self._m_step_ssl(responses)

            if self._ssl_prior.adaptive and not self._fixed_loadings:
                self._ssl_prior.update_theta(self._gamma)
        else:
            self._e_step(responses, prior_mean, prior_cov_inv)
            current_elbo = self._compute_elbo(responses, prior_mean, prior_cov)
            self._elbo_history.append(current_elbo)
            self._convergence_history.append(current_elbo)
            converged = self._check_convergence(prev_elbo, current_elbo)

        sparse_loadings = self._apply_sparsity()

        effective_dim, active_factors = self._compute_effective_dimensionality(
            sparse_loadings
        )

        sparsity_pattern = np.abs(sparse_loadings) > PROB_EPSILON

        n_selected_loadings = (
            0 if self._fixed_loadings else int(np.count_nonzero(sparsity_pattern))
        )
        n_effective_params = n_items + n_selected_loadings
        log_likelihood = self._compute_log_likelihood(responses, sparse_loadings)
        bic = -2 * log_likelihood + n_effective_params * np.log(n_persons)
        n_loading_candidates = 0 if self._fixed_loadings else n_items * self.k_max
        log_model_count = (
            lgamma(n_loading_candidates + 1)
            - lgamma(n_selected_loadings + 1)
            - lgamma(n_loading_candidates - n_selected_loadings + 1)
            if n_loading_candidates
            else 0.0
        )
        ebic = bic + log_model_count

        model._parameters["discrimination"] = (
            sparse_loadings.ravel() if model.n_factors == 1 else sparse_loadings.copy()
        )
        model._parameters["difficulty"] = self._compute_difficulty_from_intercept(
            sparse_loadings
        )
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
            converged=converged,
            n_observations=n_persons,
            n_items=n_items,
            elbo_history=self._elbo_history.copy(),
        )

    def _validate_prior(
        self,
        prior_mean: NDArray[np.float64] | None,
        prior_cov: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return finite prior arrays with a positive-definite covariance."""
        try:
            mean = (
                np.zeros(self.k_max, dtype=np.float64)
                if prior_mean is None
                else np.asarray(prior_mean, dtype=np.float64)
            )
            cov = (
                np.eye(self.k_max, dtype=np.float64)
                if prior_cov is None
                else np.asarray(prior_cov, dtype=np.float64)
            )
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "prior_mean and prior_cov must be numeric arrays"
            ) from exc

        if mean.shape != (self.k_max,):
            raise MirtValidationError(
                "prior_mean shape must match k_max",
                parameter="prior_mean",
                value=mean.shape,
                expected=f"({self.k_max},)",
            )
        if not np.all(np.isfinite(mean)):
            raise MirtValidationError(
                "prior_mean must contain only finite values",
                parameter="prior_mean",
            )
        if cov.shape != (self.k_max, self.k_max):
            raise MirtValidationError(
                "prior_cov shape must match k_max",
                parameter="prior_cov",
                value=cov.shape,
                expected=f"({self.k_max}, {self.k_max})",
            )
        if not np.all(np.isfinite(cov)):
            raise MirtValidationError(
                "prior_cov must contain only finite values",
                parameter="prior_cov",
            )
        if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
            raise MirtValidationError(
                "prior_cov must be symmetric",
                parameter="prior_cov",
            )
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise MirtValidationError(
                "prior_cov must be positive definite",
                parameter="prior_cov",
            ) from exc

        return mean.copy(), cov.copy()

    def _initialize_parameters(
        self,
        responses: NDArray[np.int_],
        n_items: int,
        *,
        fixed_loadings: bool = False,
    ) -> None:
        """Initialize loadings and intercepts."""
        valid = responses >= 0
        counts = np.sum(valid, axis=0)
        totals = np.sum(np.where(valid, responses, 0.0), axis=0)
        p = np.divide(
            totals,
            counts,
            out=np.full(n_items, 0.5, dtype=np.float64),
            where=counts > 0,
        )
        p = np.clip(p, PROB_CLIP_MIN, PROB_CLIP_MAX)

        self._intercepts = np.log(p / (1 - p))

        if fixed_loadings:
            self._loadings = np.ones((n_items, self.k_max), dtype=np.float64)
            self._gamma = np.ones((n_items, self.k_max), dtype=np.float64)
        else:
            rng = np.random.default_rng(42)
            self._loadings = rng.normal(0, 0.5, (n_items, self.k_max))
            self._loadings[counts == 0] = 0.0
            self._gamma = self._ssl_prior.compute_posterior_inclusion(self._loadings)

    def _compute_difficulty_from_intercept(
        self,
        loadings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Convert intercepts back to difficulty parameters."""
        if loadings is None:
            loadings = self._loadings
        a_sum = np.sum(loadings, axis=1)
        return np.divide(
            -self._intercepts,
            a_sum,
            out=np.zeros_like(self._intercepts),
            where=np.abs(a_sum) > PROB_EPSILON,
        )

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
        prior_mean: NDArray[np.float64],
        prior_cov_inv: NDArray[np.float64],
    ) -> None:
        """E-step: update variational parameters."""
        valid_mask = responses >= 0
        prior_natural_mean = prior_cov_inv @ prior_mean

        for _ in range(self.n_inner_iter):
            lam = self._lambda_jj(self._xi)
            weights = np.where(valid_mask, 2.0 * lam, 0.0)
            precision = prior_cov_inv + np.einsum(
                "ij,jf,jg->ifg",
                weights,
                self._loadings,
                self._loadings,
                optimize=True,
            )
            self._sigma = np.linalg.inv(precision)

            coeffs = np.where(
                valid_mask,
                responses - 0.5 - 2.0 * lam * self._intercepts,
                0.0,
            )
            natural_mean = coeffs @ self._loadings + prior_natural_mean
            self._mu = np.einsum(
                "ifg,ig->if",
                self._sigma,
                natural_mean,
                optimize=True,
            )

            eta_mean = self._mu @ self._loadings.T + self._intercepts
            eta_variance = np.einsum(
                "jf,ifg,jg->ij",
                self._loadings,
                self._sigma,
                self._loadings,
                optimize=True,
            )
            updated_xi = np.sqrt(np.maximum(eta_variance + eta_mean**2, PROB_EPSILON))
            self._xi = np.where(valid_mask, updated_xi, self._xi)

    def _m_step_ssl(
        self,
        responses: NDArray[np.int_],
    ) -> None:
        """M-step with spike-slab LASSO penalty."""
        valid_mask = responses >= 0
        lam = self._lambda_jj(self._xi)

        if self._fixed_loadings:
            self._loadings.fill(1.0)
            self._gamma.fill(1.0)
        else:
            second_moments = self._sigma + np.einsum(
                "if,ig->ifg",
                self._mu,
                self._mu,
                optimize=True,
            )
            weights = np.where(valid_mask, 2.0 * lam, 0.0)
            curvature = np.einsum(
                "ij,ifg->jfg",
                weights,
                second_moments,
                optimize=True,
            )
            curvature += REGULARIZATION_EPSILON * np.eye(self.k_max)

            coefficients = np.where(
                valid_mask,
                responses - 0.5 - 2.0 * lam * self._intercepts,
                0.0,
            )
            score = np.einsum(
                "ij,if->jf",
                coefficients,
                self._mu,
                optimize=True,
            )
            unpenalized = np.linalg.solve(curvature, score[..., None])[..., 0]
            self._gamma = self._ssl_prior.compute_posterior_inclusion(unpenalized)

            penalty_rate = self._ssl_prior.compute_effective_penalty(self._gamma)
            curvature_diag = np.diagonal(curvature, axis1=1, axis2=2)
            threshold = penalty_rate / (curvature_diag + PROB_EPSILON)
            self._loadings = self._ssl_prior.soft_threshold(
                unpenalized,
                threshold,
            )

        linear_terms = self._mu @ self._loadings.T
        intercept_score = np.sum(
            np.where(
                valid_mask,
                responses - 0.5 - 2.0 * lam * linear_terms,
                0.0,
            ),
            axis=0,
        )
        intercept_curvature = np.sum(np.where(valid_mask, 2.0 * lam, 0.0), axis=0)
        self._intercepts = np.divide(
            intercept_score,
            intercept_curvature,
            out=self._intercepts.copy(),
            where=intercept_curvature > PROB_EPSILON,
        )
        self._intercepts = np.clip(self._intercepts, -10.0, 10.0)

    def _compute_elbo(
        self,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
    ) -> float:
        """Compute evidence lower bound."""
        valid_mask = responses >= 0
        lam = self._lambda_jj(self._xi)
        eta_mean = self._mu @ self._loadings.T + self._intercepts
        eta_variance = np.einsum(
            "jf,ifg,jg->ij",
            self._loadings,
            self._sigma,
            self._loadings,
            optimize=True,
        )
        eta_second = eta_variance + eta_mean**2
        likelihood_terms = (
            -np.logaddexp(0.0, -self._xi)
            + (responses - 0.5) * eta_mean
            - 0.5 * self._xi
            - lam * (eta_second - self._xi**2)
        )
        expected_log_likelihood = float(np.sum(likelihood_terms[valid_mask]))

        prior_cov_inv = np.linalg.solve(prior_cov, np.eye(self.k_max))
        log_det_prior = float(np.linalg.slogdet(prior_cov)[1])
        log_det_q = np.linalg.slogdet(self._sigma)[1]
        diff = self._mu - prior_mean
        kl_mean = 0.5 * np.einsum(
            "if,fg,ig->i",
            diff,
            prior_cov_inv,
            diff,
            optimize=True,
        )
        kl_trace = 0.5 * np.einsum(
            "fg,igf->i",
            prior_cov_inv,
            self._sigma,
            optimize=True,
        )
        kl = kl_mean + kl_trace + 0.5 * (log_det_prior - log_det_q) - 0.5 * self.k_max

        ssl_log_prior = (
            0.0
            if self._fixed_loadings
            else float(np.sum(self._ssl_prior.log_pdf(self._loadings)))
        )
        return expected_log_likelihood - float(np.sum(kl)) + ssl_log_prior

    def _compute_log_likelihood(
        self,
        responses: NDArray[np.int_],
        loadings: NDArray[np.float64],
    ) -> float:
        """Return stable conditional log-likelihood at posterior means."""
        eta = self._mu @ loadings.T + self._intercepts
        terms = responses * eta - np.logaddexp(0.0, eta)
        return float(np.sum(terms[responses >= 0]))

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
        nonzero_per_factor = np.sum(np.abs(sparse_loadings) > PROB_EPSILON, axis=0)
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

    @property
    def variational_means(self) -> NDArray[np.float64] | None:
        """Return posterior mean ability estimates after fitting."""
        return self._mu.copy() if self._mu is not None else None

    @property
    def variational_covariances(self) -> NDArray[np.float64] | None:
        """Return posterior ability covariance estimates after fitting."""
        return self._sigma.copy() if self._sigma is not None else None
