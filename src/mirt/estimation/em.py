from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from mirt._core import sigmoid
from mirt._gpu_backend import (
    GPU_AVAILABLE,
    compute_log_likelihoods_2pl_gpu,
    compute_log_likelihoods_3pl_gpu,
    compute_log_likelihoods_gpcm_gpu,
    compute_log_likelihoods_grm_gpu,
)
from mirt.estimation.base import BaseEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.estimation.latent_density import LatentDensity
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


class EMEstimator(BaseEstimator):
    def __init__(
        self,
        n_quadpts: int = 21,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        latent_density: LatentDensity
        | Literal["gaussian", "empirical", "davidian", "mixture"]
        | None = None,
        prob_epsilon: float = 1e-10,
        item_optim_maxiter: int = 50,
        item_optim_ftol: float = 1e-6,
        se_step_size: float = 1e-5,
        n_jobs: int = 1,
        use_gpu: bool | Literal["auto"] = "auto",
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        if n_quadpts < 5:
            raise ValueError("n_quadpts should be at least 5")

        self.n_quadpts = n_quadpts
        self.prob_epsilon = prob_epsilon
        self.item_optim_maxiter = item_optim_maxiter
        self.item_optim_ftol = item_optim_ftol
        self.se_step_size = se_step_size
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self._quadrature: GaussHermiteQuadrature | None = None
        self._latent_density_spec = latent_density
        self._latent_density: LatentDensity | None = None

    @property
    def _should_use_gpu(self) -> bool:
        """Determine if GPU should be used based on settings and availability."""
        if self.use_gpu == "auto":
            return GPU_AVAILABLE
        return bool(self.use_gpu) and GPU_AVAILABLE

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
    ) -> FitResult:
        from mirt.estimation.latent_density import GaussianDensity, create_density
        from mirt.results.fit_result import FitResult

        responses = self._validate_responses(responses, model.n_items)
        n_persons = responses.shape[0]

        self._quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=model.n_factors,
        )

        if prior_mean is None:
            prior_mean = np.zeros(model.n_factors)
        if prior_cov is None:
            prior_cov = np.eye(model.n_factors)

        if self._latent_density_spec is None:
            self._latent_density = GaussianDensity(
                mean=prior_mean,
                cov=prior_cov,
                n_dimensions=model.n_factors,
            )
        elif isinstance(self._latent_density_spec, str):
            self._latent_density = create_density(
                self._latent_density_spec,
                n_dimensions=model.n_factors,
            )
        else:
            self._latent_density = self._latent_density_spec

        if not model._is_fitted:
            model._initialize_parameters()

        valid_masks = [responses[:, j] >= 0 for j in range(model.n_items)]

        self._convergence_history = []
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            posterior_weights, marginal_ll = self._e_step(model, responses)

            current_ll = np.sum(np.log(marginal_ll + 1e-300))
            self._convergence_history.append(current_ll)

            self._log_iteration(iteration, current_ll)

            if self._check_convergence(prev_ll, current_ll):
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_ll = current_ll

            self._m_step(model, responses, posterior_weights, valid_masks)

            n_k = posterior_weights.sum(axis=0)
            self._latent_density.update(self._quadrature.nodes, n_k)

        model._is_fitted = True

        standard_errors = self._compute_standard_errors(
            model, responses, posterior_weights
        )

        n_params = model.n_parameters + self._latent_density.n_parameters
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

    def _e_step(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights

        log_likelihoods = self._compute_log_likelihoods(model, responses, quad_points)

        log_prior = self._latent_density.log_density(quad_points)

        log_joint = log_likelihoods + log_prior[None, :] + np.log(quad_weights)[None, :]

        log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
        log_posterior = log_joint - log_marginal

        posterior_weights = np.exp(log_posterior)
        marginal_ll = np.exp(log_marginal.ravel())

        return posterior_weights, marginal_ll

    def _compute_log_likelihoods(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        quad_points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihoods, using GPU if available and appropriate."""
        if self._should_use_gpu and model.n_factors == 1:
            return self._compute_log_likelihoods_gpu(model, responses, quad_points)

        if hasattr(model, "log_likelihood_batch"):
            return model.log_likelihood_batch(responses, quad_points)

        return self._compute_log_likelihoods_python(model, responses, quad_points)

    def _compute_log_likelihoods_python(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        quad_points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Pure-Python log-likelihood fallback used by CPU/GPU paths."""
        n_persons = responses.shape[0]
        n_quad = quad_points.shape[0]
        log_likelihoods = np.zeros((n_persons, n_quad))
        for q in range(n_quad):
            theta_q = quad_points[q : q + 1]
            log_likelihoods[:, q] = model.log_likelihood(responses, theta_q)
        return log_likelihoods

    def _compute_log_likelihoods_gpu(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        quad_points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihoods using GPU acceleration."""
        params = model.parameters

        if model.model_name == "2PL":
            discrimination = params["discrimination"]
            difficulty = params["difficulty"]
            return compute_log_likelihoods_2pl_gpu(
                responses,
                quad_points.ravel(),
                discrimination,
                difficulty,
            )

        if model.model_name == "3PL":
            discrimination = params["discrimination"]
            difficulty = params["difficulty"]
            guessing = params["guessing"]
            return compute_log_likelihoods_3pl_gpu(
                responses,
                quad_points.ravel(),
                discrimination,
                difficulty,
                guessing,
            )

        if model.model_name == "GRM":
            discrimination = params["discrimination"]
            thresholds = params["thresholds"]
            return compute_log_likelihoods_grm_gpu(
                responses,
                quad_points.ravel(),
                discrimination,
                thresholds,
            )

        if model.model_name == "GPCM":
            discrimination = params["discrimination"]
            thresholds = params["thresholds"]
            return compute_log_likelihoods_gpcm_gpu(
                responses,
                quad_points.ravel(),
                discrimination,
                thresholds,
            )

        if hasattr(model, "log_likelihood_batch"):
            return model.log_likelihood_batch(responses, quad_points)

        return self._compute_log_likelihoods_python(model, responses, quad_points)

    def _m_step(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        valid_masks: list[NDArray[np.bool_]] | None = None,
    ) -> None:
        import os
        from concurrent.futures import ThreadPoolExecutor

        quad_points = self._quadrature.nodes
        n_items = model.n_items
        n_quad = len(quad_points)

        n_k = posterior_weights.sum(axis=0)

        if valid_masks is None:
            valid_masks = [responses[:, j] >= 0 for j in range(n_items)]

        if not model.is_polytomous:
            r_k_all = np.zeros((n_items, n_quad))
            n_k_valid_all = np.zeros((n_items, n_quad))
            for j in range(n_items):
                valid = valid_masks[j]
                item_resp = responses[:, j]
                r_k_all[j] = np.sum(
                    item_resp[valid, None] * posterior_weights[valid, :], axis=0
                )
                n_k_valid_all[j] = np.sum(posterior_weights[valid], axis=0)
        else:
            r_k_all = None
            n_k_valid_all = None

        n_jobs = self.n_jobs
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        if n_jobs == 1:
            for item_idx in range(n_items):
                r_k = r_k_all[item_idx] if r_k_all is not None else None
                n_k_valid = (
                    n_k_valid_all[item_idx] if n_k_valid_all is not None else None
                )
                self._optimize_item(
                    model,
                    item_idx,
                    responses,
                    posterior_weights,
                    quad_points,
                    n_k,
                    valid_masks[item_idx],
                    r_k,
                    n_k_valid,
                )
        else:

            def optimize_single_item(item_idx):
                r_k = r_k_all[item_idx] if r_k_all is not None else None
                n_k_valid = (
                    n_k_valid_all[item_idx] if n_k_valid_all is not None else None
                )
                return item_idx, self._optimize_item_return(
                    model,
                    item_idx,
                    responses,
                    posterior_weights,
                    quad_points,
                    n_k,
                    valid_masks[item_idx],
                    r_k,
                    n_k_valid,
                )

            with ThreadPoolExecutor(max_workers=min(n_jobs, n_items)) as executor:
                results = list(executor.map(optimize_single_item, range(n_items)))

            for item_idx, optimal_params in results:
                self._set_item_params(model, item_idx, optimal_params)

    def _supports_analytic_dichotomous_gradient(self, model: BaseItemModel) -> bool:
        """Return whether analytic item gradients are implemented for this model."""
        return model.model_name in {"1PL", "2PL", "3PL", "4PL"}

    def _neg_expected_loglik_with_grad_dichotomous(
        self,
        model: BaseItemModel,
        item_idx: int,
        quad_points: NDArray[np.float64],
        n_k_valid: NDArray[np.float64],
        r_k: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        """Compute negative expected log-likelihood and analytic gradient."""
        eps = self.prob_epsilon
        theta = quad_points
        theta_1d = theta[:, 0]
        model_name = model.model_name

        if model_name == "1PL":
            a = float(model._parameters["discrimination"][item_idx])
            b = float(params[0])

            z = a * (theta_1d - b)
            p_star = sigmoid(z)
            p = np.clip(p_star, eps, 1 - eps)

            ll = np.sum(r_k * np.log(p) + (n_k_valid - r_k) * np.log(1 - p))
            score = r_k / p - (n_k_valid - r_k) / (1 - p)
            dp_dz = p_star * (1 - p_star)

            grad_b = np.sum(score * dp_dz * (-a))
            grad = np.array([grad_b], dtype=np.float64)
            return -float(ll), -grad

        if model_name == "2PL":
            if model.n_factors == 1:
                a = float(params[0])
                b = float(params[1])
                z = a * (theta_1d - b)
                p_star = sigmoid(z)
                p = np.clip(p_star, eps, 1 - eps)

                ll = np.sum(r_k * np.log(p) + (n_k_valid - r_k) * np.log(1 - p))
                score = r_k / p - (n_k_valid - r_k) / (1 - p)
                dp_dz = p_star * (1 - p_star)

                common = score * dp_dz
                grad_a = np.sum(common * (theta_1d - b))
                grad_b = np.sum(common * (-a))
                grad = np.array([grad_a, grad_b], dtype=np.float64)
                return -float(ll), -grad

            a_vec = np.asarray(params[:-1], dtype=np.float64)
            b = float(params[-1])

            z = theta @ a_vec - np.sum(a_vec) * b
            p_star = sigmoid(z)
            p = np.clip(p_star, eps, 1 - eps)

            ll = np.sum(r_k * np.log(p) + (n_k_valid - r_k) * np.log(1 - p))
            score = r_k / p - (n_k_valid - r_k) / (1 - p)
            dp_dz = p_star * (1 - p_star)

            common = score * dp_dz
            grad_a = (theta - b).T @ common
            grad_b = np.sum(common * (-np.sum(a_vec)))

            grad = np.concatenate([grad_a, np.array([grad_b], dtype=np.float64)])
            return -float(ll), -grad

        if model_name == "3PL":
            a = float(params[0])
            b = float(params[1])
            c = float(params[2])

            z = a * (theta_1d - b)
            p_star = sigmoid(z)
            p = c + (1.0 - c) * p_star
            p = np.clip(p, eps, 1 - eps)

            ll = np.sum(r_k * np.log(p) + (n_k_valid - r_k) * np.log(1 - p))
            score = r_k / p - (n_k_valid - r_k) / (1 - p)
            dp_dz = (1.0 - c) * p_star * (1.0 - p_star)
            common = score * dp_dz

            grad_a = np.sum(common * (theta_1d - b))
            grad_b = np.sum(common * (-a))
            grad_c = np.sum(score * (1.0 - p_star))

            grad = np.array([grad_a, grad_b, grad_c], dtype=np.float64)
            return -float(ll), -grad

        if model_name == "4PL":
            a = float(params[0])
            b = float(params[1])
            c = float(params[2])
            d = float(params[3])

            z = a * (theta_1d - b)
            p_star = sigmoid(z)
            p = c + (d - c) * p_star
            p = np.clip(p, eps, 1 - eps)

            ll = np.sum(r_k * np.log(p) + (n_k_valid - r_k) * np.log(1 - p))
            score = r_k / p - (n_k_valid - r_k) / (1 - p)
            dp_dz = (d - c) * p_star * (1.0 - p_star)
            common = score * dp_dz

            grad_a = np.sum(common * (theta_1d - b))
            grad_b = np.sum(common * (-a))
            grad_c = np.sum(score * (1.0 - p_star))
            grad_d = np.sum(score * p_star)

            grad = np.array([grad_a, grad_b, grad_c, grad_d], dtype=np.float64)
            return -float(ll), -grad

        raise ValueError(f"Analytic gradient not implemented for {model_name}")

    def _optimize_item_params(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        quad_points: NDArray[np.float64],
        n_k: NDArray[np.float64],
        valid_mask: NDArray[np.bool_] | None = None,
        r_k: NDArray[np.float64] | None = None,
        n_k_valid: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Optimize item parameters and return optimal parameter vector."""
        item_responses = responses[:, item_idx]
        if valid_mask is None:
            valid_mask = item_responses >= 0

        if n_k_valid is None:
            n_k_valid = np.sum(posterior_weights[valid_mask], axis=0)

        current_params, bounds = self._get_item_params_and_bounds(model, item_idx)

        if model.is_polytomous:
            n_categories = model._n_categories[item_idx]
            n_quad = len(n_k)

            r_kc = np.zeros((n_quad, n_categories))
            for c in range(n_categories):
                cat_mask = valid_mask & (item_responses == c)
                r_kc[:, c] = np.sum(posterior_weights[cat_mask, :], axis=0)

            eps = self.prob_epsilon

            def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
                self._set_item_params(model, item_idx, params)

                probs = model.probability(quad_points, item_idx)
                probs = np.clip(probs, eps, 1 - eps)

                ll = np.sum(r_kc * np.log(probs))

                return -ll

            result = minimize(
                neg_expected_log_likelihood,
                x0=current_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": self.item_optim_maxiter,
                    "ftol": self.item_optim_ftol,
                },
            )
            return result.x

        if r_k is None:
            r_k = np.sum(
                item_responses[valid_mask, None] * posterior_weights[valid_mask, :],
                axis=0,
            )

        if self._supports_analytic_dichotomous_gradient(model):

            def neg_ll_and_grad(
                params: NDArray[np.float64],
            ) -> tuple[float, NDArray[np.float64]]:
                return self._neg_expected_loglik_with_grad_dichotomous(
                    model,
                    item_idx,
                    quad_points,
                    n_k_valid,
                    r_k,
                    params,
                )

            result = minimize(
                neg_ll_and_grad,
                x0=current_params,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "maxiter": self.item_optim_maxiter,
                    "ftol": self.item_optim_ftol,
                },
            )
            return result.x

        eps = self.prob_epsilon

        def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
            self._set_item_params(model, item_idx, params)

            probs = model.probability(quad_points, item_idx)
            probs = np.clip(probs, eps, 1 - eps)

            ll = np.sum(r_k * np.log(probs) + (n_k_valid - r_k) * np.log(1 - probs))

            return -ll

        result = minimize(
            neg_expected_log_likelihood,
            x0=current_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.item_optim_maxiter, "ftol": self.item_optim_ftol},
        )

        return result.x

    def _optimize_item(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        quad_points: NDArray[np.float64],
        n_k: NDArray[np.float64],
        valid_mask: NDArray[np.bool_] | None = None,
        r_k: NDArray[np.float64] | None = None,
        n_k_valid: NDArray[np.float64] | None = None,
    ) -> None:
        optimal = self._optimize_item_params(
            model,
            item_idx,
            responses,
            posterior_weights,
            quad_points,
            n_k,
            valid_mask,
            r_k,
            n_k_valid,
        )
        self._set_item_params(model, item_idx, optimal)

    def _optimize_item_return(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        quad_points: NDArray[np.float64],
        n_k: NDArray[np.float64],
        valid_mask: NDArray[np.bool_] | None = None,
        r_k: NDArray[np.float64] | None = None,
        n_k_valid: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Optimize item parameters and return the result (for parallel execution)."""
        return self._optimize_item_params(
            model,
            item_idx,
            responses,
            posterior_weights,
            quad_points,
            n_k,
            valid_mask,
            r_k,
            n_k_valid,
        )

    def _compute_standard_errors(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        standard_errors: dict[str, NDArray[np.float64]] = {}
        params = model.parameters

        for name, values in params.items():
            if name == "discrimination" and model.model_name == "1PL":
                standard_errors[name] = np.zeros_like(values)
                continue

            se = np.zeros_like(values)

            for item_idx in range(model.n_items):
                item_se = self._compute_item_se(
                    model, item_idx, name, responses, posterior_weights
                )
                if values.ndim == 1:
                    se[item_idx] = item_se
                else:
                    se[item_idx] = item_se

            standard_errors[name] = se

        return standard_errors

    def _compute_item_se(
        self,
        model: BaseItemModel,
        item_idx: int,
        param_name: str,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
    ) -> float | NDArray[np.float64]:
        quad_points = self._quadrature.nodes
        item_responses = responses[:, item_idx]
        valid_mask = item_responses >= 0

        values = model.parameters[param_name]
        if values.ndim == 1:
            current = float(values[item_idx])
            is_scalar = True
        else:
            current = values[item_idx].copy()
            is_scalar = False

        n_k_valid = np.sum(posterior_weights[valid_mask], axis=0)

        eps = self.prob_epsilon

        if model.is_polytomous:
            n_categories = model._n_categories[item_idx]
            n_quad = len(n_k_valid)
            r_kc = np.zeros((n_quad, n_categories))
            for c in range(n_categories):
                cat_mask = valid_mask & (item_responses == c)
                r_kc[:, c] = np.sum(posterior_weights[cat_mask, :], axis=0)

            def log_likelihood(param_val: float | NDArray[np.float64]) -> float:
                model.set_item_parameter(item_idx, param_name, param_val)

                probs = model.probability(quad_points, item_idx)
                probs = np.clip(probs, eps, 1 - eps)

                ll = float(np.sum(r_kc * np.log(probs)))

                model.set_item_parameter(item_idx, param_name, current)
                return ll

        else:
            r_k = np.sum(
                item_responses[valid_mask, None] * posterior_weights[valid_mask, :],
                axis=0,
            )

            def log_likelihood(param_val: float | NDArray[np.float64]) -> float:
                model.set_item_parameter(item_idx, param_name, param_val)

                probs = model.probability(quad_points, item_idx)
                probs = np.clip(probs, eps, 1 - eps)

                ll = float(
                    np.sum(r_k * np.log(probs) + (n_k_valid - r_k) * np.log(1 - probs))
                )

                model.set_item_parameter(item_idx, param_name, current)
                return ll

        h = self.se_step_size
        ll_center = log_likelihood(current)

        if is_scalar:
            ll_plus = log_likelihood(current + h)
            ll_minus = log_likelihood(current - h)

            hessian = (ll_plus - 2 * ll_center + ll_minus) / (h**2)

            if hessian < 0:
                se = np.sqrt(-1.0 / hessian)
            else:
                se = np.nan

            return se
        else:
            n_params = len(current)
            se = np.zeros(n_params)

            for i in range(n_params):
                param_plus = current.copy()
                param_plus[i] += h
                param_minus = current.copy()
                param_minus[i] -= h

                ll_plus = log_likelihood(param_plus)
                ll_minus = log_likelihood(param_minus)

                hessian = (ll_plus - 2 * ll_center + ll_minus) / (h**2)

                if hessian < 0:
                    se[i] = np.sqrt(-1.0 / hessian)
                else:
                    se[i] = np.nan

            return se

    @staticmethod
    def _log_multivariate_normal(
        x: NDArray[np.float64],
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        n, d = x.shape
        diff = x - mean

        try:
            L = np.linalg.cholesky(cov)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            solve = np.linalg.solve(L, diff.T)
            maha = np.sum(solve**2, axis=0)
        except np.linalg.LinAlgError:
            sign, log_det = np.linalg.slogdet(cov)
            cov_inv = np.linalg.pinv(cov)
            maha = np.sum(diff @ cov_inv * diff, axis=1)

        log_norm = -0.5 * (d * np.log(2 * np.pi) + log_det)
        return log_norm - 0.5 * maha
