from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt._rust_backend import (
    gvem_compute_elbo as _rust_gvem_compute_elbo,
)
from mirt._rust_backend import (
    gvem_e_step as _rust_gvem_e_step,
)
from mirt._rust_backend import (
    gvem_m_step as _rust_gvem_m_step,
)
from mirt.constants import PROB_EPSILON, REGULARIZATION_EPSILON
from mirt.estimation.base import BaseEstimator

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


class GVEMEstimator(BaseEstimator):
    """Gaussian Variational EM estimator for multidimensional IRT models.

    Uses variational inference with the Jaakkola-Jordan bound on the logistic
    function to achieve closed-form E-step updates. This makes GVEM highly
    efficient for high-dimensional models where quadrature-based EM becomes
    computationally infeasible.

    Parameters
    ----------
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance for ELBO change.
    verbose : bool
        Whether to print progress during fitting.
    n_inner_iter : int
        Number of inner iterations for variational parameter updates per EM step.
    se_step_size : float
        Step size for numerical differentiation when computing standard errors.

    Notes
    -----
    GVEM maintains a Gaussian variational distribution q_i(theta_i) = N(mu_i, Sigma_i)
    for each person's latent abilities. The Jaakkola-Jordan bound provides a
    quadratic lower bound on the logistic log-likelihood, enabling closed-form
    updates for the variational parameters.

    Currently supports 2PL models (unidimensional and multidimensional).

    References
    ----------
    Cho, A. E., Wang, C., Zhang, X., & Xu, G. (2021). Gaussian variational
        estimation for multidimensional item response theory. British Journal
        of Mathematical and Statistical Psychology, 74, 52-85.

    Jaakkola, T. S., & Jordan, M. I. (2000). Bayesian parameter estimation
        via variational methods. Statistics and Computing, 10(1), 25-37.
    """

    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        n_inner_iter: int = 3,
        se_step_size: float = 1e-5,
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        if n_inner_iter < 1:
            raise ValueError("n_inner_iter must be at least 1")

        self.n_inner_iter = n_inner_iter
        self.se_step_size = se_step_size

        self._mu: NDArray[np.float64] | None = None
        self._sigma: NDArray[np.float64] | None = None
        self._xi: NDArray[np.float64] | None = None
        self._elbo_history: list[float] = []

        self._slopes: NDArray[np.float64] | None = None
        self._intercepts: NDArray[np.float64] | None = None

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
    ) -> FitResult:
        """Fit model using Gaussian Variational EM algorithm.

        Parameters
        ----------
        model : BaseItemModel
            IRT model to fit. Currently supports 2PL models.
        responses : ndarray of shape (n_persons, n_items)
            Response matrix with -1 indicating missing responses.
        prior_mean : ndarray of shape (n_factors,), optional
            Prior mean for latent abilities. Defaults to zeros.
        prior_cov : ndarray of shape (n_factors, n_factors), optional
            Prior covariance for latent abilities. Defaults to identity.

        Returns
        -------
        FitResult
            Fitted model with estimates, ELBO, and diagnostics.

        Raises
        ------
        ValueError
            If model type is not supported.
        """
        from mirt.results.fit_result import FitResult

        if model.model_name not in ("2PL", "1PL"):
            raise ValueError(
                f"GVEMEstimator currently only supports 2PL models, got {model.model_name}"
            )

        responses = self._validate_responses(responses, model.n_items)
        n_persons = responses.shape[0]
        n_items = model.n_items
        n_factors = model.n_factors

        if prior_mean is None:
            prior_mean = np.zeros(n_factors)
        if prior_cov is None:
            prior_cov = np.eye(n_factors)

        if not model._is_fitted:
            model._initialize_parameters()

        self._convert_to_slope_intercept(model)

        self._initialize_variational_params(n_persons, n_factors, n_items)

        self._convergence_history = []
        self._elbo_history = []
        prev_elbo = -np.inf

        for iteration in range(self.max_iter):
            self._e_step(model, responses, prior_cov)

            current_elbo = self._compute_elbo(model, responses, prior_mean, prior_cov)
            self._elbo_history.append(current_elbo)
            self._convergence_history.append(current_elbo)

            self._log_iteration(iteration, current_elbo, elbo=current_elbo)

            if self._check_convergence(prev_elbo, current_elbo):
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_elbo = current_elbo

            self._m_step(model, responses)

        self._convert_from_slope_intercept(model)

        model._is_fitted = True

        standard_errors = self._compute_standard_errors(model, responses)

        n_params = model.n_parameters
        aic = self._compute_aic(current_elbo, n_params)
        bic = self._compute_bic(current_elbo, n_params, n_persons)

        return FitResult(
            model=model,
            log_likelihood=current_elbo,
            n_iterations=iteration + 1,
            converged=iteration < self.max_iter - 1,
            standard_errors=standard_errors,
            aic=aic,
            bic=bic,
            n_observations=n_persons,
            n_parameters=n_params,
        )

    def _convert_to_slope_intercept(self, model: BaseItemModel) -> None:
        """Convert model parameters from discrimination-difficulty to slope-intercept form."""
        a = model.parameters["discrimination"]
        b = model.parameters["difficulty"]

        if model.n_factors == 1:
            self._slopes = a.reshape(-1, 1).copy()
            self._intercepts = (-a * b).copy()
        else:
            self._slopes = a.copy()
            self._intercepts = (-np.sum(a, axis=1) * b).copy()

    def _convert_from_slope_intercept(self, model: BaseItemModel) -> None:
        """Convert parameters from slope-intercept back to discrimination-difficulty form."""
        if model.n_factors == 1:
            a = self._slopes.ravel()
            d = self._intercepts
            b = -d / (a + PROB_EPSILON)

            model._parameters["discrimination"] = a
            model._parameters["difficulty"] = b
        else:
            a = self._slopes
            d = self._intercepts
            a_sum = np.sum(a, axis=1)
            b = -d / (a_sum + PROB_EPSILON)

            model._parameters["discrimination"] = a
            model._parameters["difficulty"] = b

    def _initialize_variational_params(
        self,
        n_persons: int,
        n_factors: int,
        n_items: int,
    ) -> None:
        """Initialize variational parameters."""
        self._mu = np.zeros((n_persons, n_factors))
        self._sigma = np.tile(np.eye(n_factors), (n_persons, 1, 1))
        self._xi = np.ones((n_persons, n_items))

    @staticmethod
    def _lambda(xi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the Jaakkola-Jordan lambda function.

        lambda(xi) = tanh(xi/2) / (4*xi)

        This is numerically stable for xi near 0 where the limit is 1/8.
        """
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
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_cov: NDArray[np.float64],
    ) -> None:
        """E-step: update variational parameters with closed-form updates."""
        prior_cov_inv = np.linalg.inv(prior_cov)

        rust_result = _rust_gvem_e_step(
            responses,
            self._slopes,
            self._intercepts,
            prior_cov_inv,
            self._mu,
            self._sigma,
            self._xi,
            self.n_inner_iter,
        )

        if rust_result is not None:
            self._mu, self._sigma, self._xi = rust_result
            return

        self._e_step_python(model, responses, prior_cov_inv)

    def _e_step_python(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_cov_inv: NDArray[np.float64],
    ) -> None:
        """Python fallback for E-step."""
        n_persons, n_items = responses.shape

        valid_mask = responses >= 0

        for _ in range(self.n_inner_iter):
            lam = self._lambda(self._xi)

            for i in range(n_persons):
                valid_items = valid_mask[i]
                if not valid_items.any():
                    continue

                a_valid = self._slopes[valid_items]
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

                    a_j = self._slopes[j]
                    d_j = self._intercepts[j]

                    quad_term = a_j @ second_moment @ a_j
                    linear_term = 2 * d_j * (a_j @ mu_i)
                    const_term = d_j**2

                    self._xi[i, j] = np.sqrt(quad_term + linear_term + const_term)

    def _m_step(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> None:
        """M-step: update item parameters with closed-form updates."""
        rust_result = _rust_gvem_m_step(
            responses,
            self._mu,
            self._sigma,
            self._xi,
            self._slopes,
            self._intercepts,
        )

        if rust_result is not None:
            self._slopes, self._intercepts = rust_result
            if model.model_name == "1PL":
                self._slopes[:] = 1.0
            return

        self._m_step_python(model, responses)

    def _m_step_python(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> None:
        """Python fallback for M-step."""
        n_persons, n_items = responses.shape
        n_factors = model.n_factors

        valid_mask = responses >= 0

        lam = self._lambda(self._xi)

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

            A_j += REGULARIZATION_EPSILON * np.eye(n_factors)

            try:
                a_j_new = np.linalg.solve(A_j, b_j)
            except np.linalg.LinAlgError:
                a_j_new = np.linalg.lstsq(A_j, b_j, rcond=None)[0]

            if model.model_name == "1PL":
                a_j_new = np.ones(n_factors)

            self._slopes[j] = a_j_new

            linear_terms = mu_valid @ a_j_new
            d_numerator = np.sum(y_valid - 0.5 - 2 * lam_valid * linear_terms)
            d_denominator = 2 * np.sum(lam_valid)

            if d_denominator > PROB_EPSILON:
                d_j_new = d_numerator / d_denominator
            else:
                d_j_new = 0.0

            d_j_new = np.clip(d_j_new, -10.0, 10.0)
            self._intercepts[j] = d_j_new

    def _compute_elbo(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
    ) -> float:
        """Compute the Evidence Lower Bound (ELBO).

        ELBO = E_q[log p(y|theta)] + E_q[log p(theta)] - E_q[log q(theta)]
        """
        rust_result = _rust_gvem_compute_elbo(
            responses,
            self._slopes,
            self._intercepts,
            self._mu,
            self._sigma,
            self._xi,
            prior_mean,
            prior_cov,
        )

        if rust_result is not None:
            return rust_result

        return self._compute_elbo_python(model, responses, prior_mean, prior_cov)

    def _compute_elbo_python(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
    ) -> float:
        """Python fallback for ELBO computation."""
        n_persons, n_items = responses.shape
        n_factors = model.n_factors

        valid_mask = responses >= 0

        lam = self._lambda(self._xi)

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

                a_j = self._slopes[j]
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

            kl = kl_mean + kl_trace + kl_logdet - 0.5 * n_factors

            elbo -= kl

        return float(elbo)

    def _compute_standard_errors(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> dict[str, NDArray[np.float64]]:
        """Compute standard errors using numerical differentiation of ELBO."""
        standard_errors: dict[str, NDArray[np.float64]] = {}

        for name, values in model.parameters.items():
            if name == "discrimination" and model.model_name == "1PL":
                standard_errors[name] = np.zeros_like(values)
                continue

            se = np.full_like(values, np.nan, dtype=np.float64)

            h = self.se_step_size

            if values.ndim == 1:
                for item_idx in range(len(values)):
                    original = float(values[item_idx])

                    model._parameters[name][item_idx] = original + h
                    self._convert_to_slope_intercept(model)
                    elbo_plus = self._compute_elbo(
                        model,
                        responses,
                        np.zeros(model.n_factors),
                        np.eye(model.n_factors),
                    )

                    model._parameters[name][item_idx] = original - h
                    self._convert_to_slope_intercept(model)
                    elbo_minus = self._compute_elbo(
                        model,
                        responses,
                        np.zeros(model.n_factors),
                        np.eye(model.n_factors),
                    )

                    model._parameters[name][item_idx] = original
                    self._convert_to_slope_intercept(model)
                    elbo_center = self._compute_elbo(
                        model,
                        responses,
                        np.zeros(model.n_factors),
                        np.eye(model.n_factors),
                    )

                    hessian = (elbo_plus - 2 * elbo_center + elbo_minus) / (h**2)

                    if hessian < 0:
                        se[item_idx] = np.sqrt(-1.0 / hessian)

            else:
                for item_idx in range(values.shape[0]):
                    for factor_idx in range(values.shape[1]):
                        original = float(values[item_idx, factor_idx])

                        model._parameters[name][item_idx, factor_idx] = original + h
                        self._convert_to_slope_intercept(model)
                        elbo_plus = self._compute_elbo(
                            model,
                            responses,
                            np.zeros(model.n_factors),
                            np.eye(model.n_factors),
                        )

                        model._parameters[name][item_idx, factor_idx] = original - h
                        self._convert_to_slope_intercept(model)
                        elbo_minus = self._compute_elbo(
                            model,
                            responses,
                            np.zeros(model.n_factors),
                            np.eye(model.n_factors),
                        )

                        model._parameters[name][item_idx, factor_idx] = original
                        self._convert_to_slope_intercept(model)
                        elbo_center = self._compute_elbo(
                            model,
                            responses,
                            np.zeros(model.n_factors),
                            np.eye(model.n_factors),
                        )

                        hessian = (elbo_plus - 2 * elbo_center + elbo_minus) / (h**2)

                        if hessian < 0:
                            se[item_idx, factor_idx] = np.sqrt(-1.0 / hessian)

            standard_errors[name] = se

        return standard_errors

    @property
    def elbo_history(self) -> list[float]:
        """Return history of ELBO values during fitting."""
        return self._elbo_history.copy()

    @property
    def variational_means(self) -> NDArray[np.float64] | None:
        """Return variational means (person ability estimates)."""
        return self._mu.copy() if self._mu is not None else None

    @property
    def variational_covariances(self) -> NDArray[np.float64] | None:
        """Return variational covariances (uncertainty in ability estimates)."""
        return self._sigma.copy() if self._sigma is not None else None
