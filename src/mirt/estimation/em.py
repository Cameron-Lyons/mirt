from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

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
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        if n_quadpts < 5:
            raise ValueError("n_quadpts should be at least 5")

        self.n_quadpts = n_quadpts
        self._quadrature: GaussHermiteQuadrature | None = None
        self._latent_density_spec = latent_density
        self._latent_density: LatentDensity | None = None

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

            self._m_step(model, responses, posterior_weights)

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
        n_persons = responses.shape[0]
        n_quad = len(quad_weights)

        log_likelihoods = np.zeros((n_persons, n_quad))

        for q in range(n_quad):
            theta_q = np.tile(quad_points[q], (n_persons, 1))
            log_likelihoods[:, q] = model.log_likelihood(responses, theta_q)

        log_prior = self._latent_density.log_density(quad_points)

        log_joint = log_likelihoods + log_prior[None, :] + np.log(quad_weights)[None, :]

        log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
        log_posterior = log_joint - log_marginal

        posterior_weights = np.exp(log_posterior)
        marginal_ll = np.exp(log_marginal.ravel())

        return posterior_weights, marginal_ll

    def _m_step(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
    ) -> None:
        quad_points = self._quadrature.nodes
        n_items = model.n_items

        n_k = posterior_weights.sum(axis=0)

        for item_idx in range(n_items):
            self._optimize_item(
                model, item_idx, responses, posterior_weights, quad_points, n_k
            )

    def _optimize_item(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        quad_points: NDArray[np.float64],
        n_k: NDArray[np.float64],
    ) -> None:
        item_responses = responses[:, item_idx]
        valid_mask = item_responses >= 0

        n_k_valid = np.sum(posterior_weights[valid_mask], axis=0)

        current_params, bounds = self._get_item_params_and_bounds(model, item_idx)

        if model.is_polytomous:
            n_categories = model._n_categories[item_idx]
            n_quad = len(n_k)

            r_kc = np.zeros((n_quad, n_categories))
            for c in range(n_categories):
                cat_mask = valid_mask & (item_responses == c)
                r_kc[:, c] = np.sum(posterior_weights[cat_mask, :], axis=0)

            def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
                self._set_item_params(model, item_idx, params)

                probs = model.probability(quad_points, item_idx)
                probs = np.clip(probs, 1e-10, 1 - 1e-10)

                ll = np.sum(r_kc * np.log(probs))

                return -ll

        else:
            r_k = np.sum(
                item_responses[valid_mask, None] * posterior_weights[valid_mask, :],
                axis=0,
            )

            def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
                self._set_item_params(model, item_idx, params)

                probs = model.probability(quad_points, item_idx)
                probs = np.clip(probs, 1e-10, 1 - 1e-10)

                ll = np.sum(r_k * np.log(probs) + (n_k_valid - r_k) * np.log(1 - probs))

                return -ll

        result = minimize(
            neg_expected_log_likelihood,
            x0=current_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-6},
        )

        self._set_item_params(model, item_idx, result.x)

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
                probs = np.clip(probs, 1e-10, 1 - 1e-10)

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
                probs = np.clip(probs, 1e-10, 1 - 1e-10)

                ll = float(
                    np.sum(r_k * np.log(probs) + (n_k_valid - r_k) * np.log(1 - probs))
                )

                model.set_item_parameter(item_idx, param_name, current)
                return ll

        h = 1e-5
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
