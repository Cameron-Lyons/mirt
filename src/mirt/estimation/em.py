from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import xlog1py, xlogy

from mirt._backend_config import should_use_rust
from mirt._gpu_backend import (
    compute_log_likelihoods_2pl_gpu,
    compute_log_likelihoods_3pl_gpu,
    compute_log_likelihoods_gpcm_gpu,
    compute_log_likelihoods_grm_gpu,
    is_gpu_available,
)
from mirt._model_defaults import uses_builtin_model_hooks
from mirt.backends.rust._helpers import RUST_AVAILABLE
from mirt.backends.rust.estimation import em_iteration_3pl
from mirt.constants import PROB_EPSILON
from mirt.estimation._em_context import EMFitContext
from mirt.estimation._posterior import normalize_log_posterior
from mirt.estimation.base import BaseEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.se_methods import _compute_item_se_curvature
from mirt.exceptions import MirtValidationError

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
        use_rust: bool = True,
        compute_standard_errors: bool = True,
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        if n_quadpts < 5:
            raise MirtValidationError(
                "n_quadpts should be at least 5",
                parameter="n_quadpts",
                value=n_quadpts,
                expected=">= 5",
            )

        self.n_quadpts = n_quadpts
        self.prob_epsilon = prob_epsilon
        self.item_optim_maxiter = item_optim_maxiter
        self.item_optim_ftol = item_optim_ftol
        self.se_step_size = se_step_size
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.use_rust = use_rust
        if not isinstance(compute_standard_errors, (bool, np.bool_)):
            raise MirtValidationError(
                "compute_standard_errors must be a boolean",
                parameter="compute_standard_errors",
                value=compute_standard_errors,
                expected="bool",
            )
        self.compute_standard_errors = bool(compute_standard_errors)
        self._quadrature: GaussHermiteQuadrature | None = None
        self._latent_density_spec = latent_density
        self._latent_density: LatentDensity | None = None
        self._pattern_frequencies: NDArray[np.float64] | None = None
        self._fit_context: EMFitContext | None = None

    @property
    def _should_use_gpu(self) -> bool:
        """Determine if GPU should be used based on settings and availability."""
        if self.use_gpu is False:
            return False
        return is_gpu_available()

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
    ) -> FitResult:
        from mirt.estimation._patterns import supports_pattern_compression
        from mirt.estimation.latent_density import GaussianDensity, create_density

        responses = self._validate_responses(responses, model.n_items)
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

        builtin = supports_pattern_compression(model)
        with EMFitContext(
            responses,
            compress=type(self) is EMEstimator and builtin,
            native=builtin and should_use_rust(self.use_rust),
        ) as context:
            self._fit_context = context
            self._pattern_frequencies = context.frequencies
            try:
                return self._fit_prepared(model, context)
            finally:
                self._fit_context = None

    def _fit_prepared(self, model: BaseItemModel, context: EMFitContext) -> FitResult:
        from mirt.results.fit_result import FitResult

        responses = context.responses
        n_persons = context.n_observations
        frequencies = context.frequencies
        valid_masks = [context.observed[:, j] for j in range(model.n_items)]
        use_rust_3pl = self._can_use_rust_3pl(model, responses)

        self._convergence_history = []
        prev_ll = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            rust_result = None
            if use_rust_3pl:
                rust_result = self._run_rust_3pl_iteration(model, responses)
                if rust_result is None:
                    use_rust_3pl = False

            if rust_result is None:
                posterior_weights, log_marginal = self._e_step(model, responses)
                current_ll = float(
                    np.sum(log_marginal * (1.0 if frequencies is None else frequencies))
                )
            else:
                (
                    new_discrimination,
                    new_difficulty,
                    new_guessing,
                    posterior_weights,
                    current_ll,
                ) = rust_result

            self._convergence_history.append(current_ll)

            self._log_iteration(iteration, current_ll)

            if self._check_convergence(prev_ll, current_ll):
                converged = True
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_ll = current_ll

            weighted_posterior = (
                posterior_weights
                if frequencies is None
                else posterior_weights * frequencies[:, None]
            )
            if rust_result is None:
                self._m_step(model, responses, weighted_posterior, valid_masks)
            else:
                model.set_parameters(
                    discrimination=new_discrimination,
                    difficulty=new_difficulty,
                    guessing=new_guessing,
                )

            n_k = weighted_posterior.sum(axis=0)
            self._latent_density.update(self._quadrature.nodes, n_k)
        else:
            posterior_weights, log_marginal = self._e_step(model, responses)
            current_ll = float(
                np.sum(log_marginal * (1.0 if frequencies is None else frequencies))
            )
            self._convergence_history.append(current_ll)
            converged = self._check_convergence(prev_ll, current_ll)

        model._is_fitted = True

        weighted_posterior = (
            posterior_weights
            if frequencies is None
            else posterior_weights * frequencies[:, None]
        )
        standard_errors = (
            self._compute_standard_errors(model, responses, weighted_posterior)
            if self.compute_standard_errors
            else {}
        )

        n_params = model.n_parameters + self._latent_density.n_parameters
        aic = self._compute_aic(current_ll, n_params)
        bic = self._compute_bic(current_ll, n_params, n_persons)

        return FitResult(
            model=model,
            log_likelihood=current_ll,
            n_iterations=iteration + 1,
            converged=converged,
            standard_errors=standard_errors,
            aic=aic,
            bic=bic,
            n_observations=n_persons,
            n_parameters=n_params,
        )

    def _can_use_rust_3pl(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> bool:
        """Return whether the batched native 3PL iteration preserves semantics."""
        from mirt.estimation.latent_density import GaussianDensity
        from mirt.models.dichotomous import ThreeParameterLogistic

        if (
            not RUST_AVAILABLE
            or not should_use_rust(self.use_rust)
            or self._should_use_gpu
            or type(model) is not ThreeParameterLogistic
            or not uses_builtin_model_hooks(model, likelihood=True)
            or model.n_factors != 1
            or self.n_jobs != 1
            or self.prob_epsilon != PROB_EPSILON
        ):
            return False

        if (
            isinstance(self.item_optim_maxiter, (bool, np.bool_))
            or not isinstance(self.item_optim_maxiter, Integral)
            or self.item_optim_maxiter < 1
            or isinstance(self.item_optim_ftol, (bool, np.bool_))
            or not isinstance(self.item_optim_ftol, Real)
            or not np.isfinite(self.item_optim_ftol)
            or self.item_optim_ftol <= 0.0
        ):
            return False

        density = self._latent_density
        if (
            not isinstance(density, GaussianDensity)
            or density.n_dimensions != 1
            or density.estimate_mean
            or density.estimate_cov
            or not np.array_equal(density.mean, np.zeros(1))
            or not np.array_equal(density.cov, np.eye(1))
        ):
            return False

        params = model.parameters
        expected_shape = (model.n_items,)
        if any(
            name not in params or params[name].shape != expected_shape
            for name in ("discrimination", "difficulty", "guessing")
        ):
            return False

        observed = responses[responses >= 0]
        return bool(np.all((observed == 0) | (observed == 1)))

    def _run_rust_3pl_iteration(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> (
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            float,
        ]
        | None
    ):
        """Run and validate one batched native 3PL E/M iteration."""
        if self._quadrature is None:
            return None

        params = model.parameters
        try:
            result = em_iteration_3pl(
                responses,
                self._quadrature.nodes.ravel(),
                self._quadrature.weights,
                params["discrimination"],
                params["difficulty"],
                params["guessing"],
                prior_mean=0.0,
                prior_var=1.0,
                max_m_iter=int(self.item_optim_maxiter),
                m_tol=float(self.item_optim_ftol),
                disc_bounds=(0.1, 5.0),
                diff_bounds=(-6.0, 6.0),
                guess_bounds=(0.0, 0.5),
                damping_ab=0.5,
                damping_c=0.3,
                regularization=0.01,
                regularization_c=0.1,
                frequencies=self._pattern_frequencies,
            )
        except Exception:
            return None

        try:
            if result is None or len(result) != 5:
                return None

            discrimination, difficulty, guessing, posterior, log_likelihood = result
            discrimination = np.asarray(discrimination, dtype=np.float64)
            difficulty = np.asarray(difficulty, dtype=np.float64)
            guessing = np.asarray(guessing, dtype=np.float64)
            posterior = np.asarray(posterior, dtype=np.float64)
            log_likelihood = float(log_likelihood)
        except (TypeError, ValueError, OverflowError):
            return None

        parameter_shape = (model.n_items,)
        posterior_shape = (responses.shape[0], self.n_quadpts)
        if (
            discrimination.shape != parameter_shape
            or difficulty.shape != parameter_shape
            or guessing.shape != parameter_shape
            or posterior.shape != posterior_shape
            or not np.all(np.isfinite(discrimination))
            or not np.all(np.isfinite(difficulty))
            or not np.all(np.isfinite(guessing))
            or not np.all(np.isfinite(posterior))
            or not np.isfinite(log_likelihood)
            or np.any((discrimination < 0.1) | (discrimination > 5.0))
            or np.any((difficulty < -6.0) | (difficulty > 6.0))
            or np.any((guessing < 0.0) | (guessing > 0.5))
            or np.any(posterior < 0.0)
        ):
            return None

        row_sums = posterior.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-12):
            return None
        posterior = posterior / row_sums[:, None]

        return (
            discrimination.copy(),
            difficulty.copy(),
            guessing.copy(),
            posterior,
            log_likelihood,
        )

    def _e_step(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return normalized posterior weights and per-person log marginals."""
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights

        log_likelihoods = self._compute_log_likelihoods(model, responses, quad_points)

        log_prior_mass = self._latent_density.log_quadrature_mass(
            quad_points, quad_weights
        )

        # Built-in likelihoods allocate their output. Custom likelihoods may
        # return cached arrays or views, which must not be overwritten.
        from mirt.estimation._patterns import supports_pattern_compression

        if not (
            type(self) is EMEstimator
            and "_compute_log_likelihoods" not in vars(self)
            and supports_pattern_compression(model)
            and log_likelihoods.flags.writeable
        ):
            log_likelihoods = log_likelihoods.copy()
        return normalize_log_posterior(log_likelihoods, log_prior_mass)

    def _compute_log_likelihoods(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        quad_points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihoods, using GPU if available and appropriate."""
        if (
            self._should_use_gpu
            and model.n_factors == 1
            and uses_builtin_model_hooks(model, likelihood=True)
        ):
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
        context = self._fit_context

        if model.is_polytomous and should_use_rust(self.use_rust):
            from mirt.backends.rust.polytomous_mstep import try_polytomous_m_step

            if try_polytomous_m_step(
                model,
                responses,
                posterior_weights,
                quad_points,
                max_iter=self.item_optim_maxiter,
                ftol=self.item_optim_ftol,
                epsilon=self.prob_epsilon,
                n_jobs=self.n_jobs,
                context=context,
            ):
                return

        n_k = posterior_weights.sum(axis=0)

        if valid_masks is None:
            valid_masks = [responses[:, j] >= 0 for j in range(n_items)]

        if not model.is_polytomous:
            prepared = context or EMFitContext(responses)
            r_k_all, n_k_valid_all = prepared.expected_counts(posterior_weights)
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

            if context is not None:
                results = list(
                    context.executor(n_jobs).map(optimize_single_item, range(n_items))
                )
            else:
                with ThreadPoolExecutor(max_workers=min(n_jobs, n_items)) as executor:
                    results = list(executor.map(optimize_single_item, range(n_items)))

            for item_idx, optimal_params in results:
                self._set_item_params(model, item_idx, optimal_params)

    def _neg_expected_loglik_with_grad_dichotomous(
        self,
        model: BaseItemModel,
        item_idx: int,
        quad_points: NDArray[np.float64],
        n_k_valid: NDArray[np.float64],
        r_k: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        """Evaluate the clipped item objective without assuming optimizer bounds."""
        from mirt.estimation._dichotomous_objective import prepare_dichotomous_objective

        objective = prepare_dichotomous_objective(
            model, item_idx, quad_points, n_k_valid, r_k, self.prob_epsilon
        )
        if objective is None:
            raise ValueError(
                f"Analytic gradient not implemented for {type(model).__name__}"
            )
        return objective(params)

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
        r_kc: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Optimize item parameters and return optimal parameter vector."""
        item_responses = responses[:, item_idx]
        if valid_mask is None:
            valid_mask = item_responses >= 0

        current_params, bounds = self._get_item_params_and_bounds(model, item_idx)
        if not current_params.size:
            return current_params

        objective: Callable[
            [NDArray[np.float64]], float | tuple[float, NDArray[np.float64]]
        ]
        if model.is_polytomous:
            from mirt.estimation._polytomous_objective import (
                prepare_polytomous_objective,
            )

            if r_kc is None:
                context = self._fit_context
                if context is None or context.responses is not responses:
                    context = EMFitContext(responses)
                r_kc = context.expected_category_counts(
                    item_idx,
                    model.n_categories[item_idx],
                    posterior_weights,
                    valid_mask,
                )
            if not np.any(r_kc):
                return current_params
            prepared = prepare_polytomous_objective(
                model, item_idx, quad_points, r_kc, self.prob_epsilon
            )
            if prepared is None:

                def polytomous_objective(params: NDArray[np.float64]) -> float:
                    self._set_item_params(model, item_idx, params)
                    probs = np.clip(
                        model.probability(quad_points, item_idx),
                        self.prob_epsilon,
                        1 - self.prob_epsilon,
                    )
                    return -float(np.sum(xlogy(r_kc, probs)))

                objective = polytomous_objective
        else:
            if n_k_valid is None:
                n_k_valid = np.sum(posterior_weights[valid_mask], axis=0)
            if not np.any(n_k_valid):
                return current_params
            if r_k is None:
                r_k = np.sum(
                    item_responses[valid_mask, None] * posterior_weights[valid_mask, :],
                    axis=0,
                )

            from mirt.estimation._affine_objective import prepare_affine_objective
            from mirt.estimation._dichotomous_objective import (
                prepare_dichotomous_objective,
            )

            prepared = prepare_dichotomous_objective(
                model, item_idx, quad_points, n_k_valid, r_k, self.prob_epsilon, bounds
            )
            if prepared is None:
                prepared = prepare_affine_objective(
                    model,
                    item_idx,
                    quad_points,
                    n_k_valid,
                    r_k,
                    self.prob_epsilon,
                    bounds,
                )
            if prepared is None:

                def dichotomous_objective(params: NDArray[np.float64]) -> float:
                    self._set_item_params(model, item_idx, params)
                    probs = np.clip(
                        model.probability(quad_points, item_idx),
                        self.prob_epsilon,
                        1 - self.prob_epsilon,
                    )
                    return -float(
                        np.sum(xlogy(r_k, probs) + xlog1py(n_k_valid - r_k, -probs))
                    )

                objective = dichotomous_objective

        analytic = prepared is not None
        if prepared is not None:
            objective = prepared

        try:
            result = minimize(
                objective,
                x0=current_params,
                method="L-BFGS-B",
                jac=analytic,
                bounds=bounds,
                options={
                    "maxiter": self.item_optim_maxiter,
                    "ftol": self.item_optim_ftol,
                },
            )
            return result.x
        finally:
            if not analytic:
                self._set_item_params(model, item_idx, current_params)

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
        from copy import deepcopy

        # Numerical objectives update whole parameter arrays before evaluating
        # one item. Keep custom state and instance method overrides as well as
        # parameters when isolating workers from lost trial updates.
        local_model = deepcopy(model)
        return self._optimize_item_params(
            local_model,
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
        *,
        person_weights: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        from mirt.estimation._item_information import item_standard_errors

        context = self._fit_context
        if context is None or context.responses is not responses:
            context = EMFitContext(responses)
        if self.se_step_size == 1e-5:
            analytic = item_standard_errors(
                model,
                responses,
                posterior_weights,
                self._quadrature.nodes,
                self.prob_epsilon,
                person_weights=person_weights,
                context=context,
            )
            if analytic is not None:
                return analytic
        standard_errors: dict[str, NDArray[np.float64]] = {}
        params = model.parameters
        free_masks = model.free_parameter_masks
        correct = observed = None
        category_counts: dict[int, NDArray[np.float64]] = {}
        if not model.is_polytomous:
            correct, observed = context.expected_counts(
                posterior_weights, person_weights
            )

        for name, values in params.items():
            free_mask = free_masks[name]
            if not np.any(free_mask):
                standard_errors[name] = np.zeros_like(values)
                continue

            se = np.zeros_like(values)

            for item_idx in range(model.n_items):
                if not np.any(free_mask[item_idx]):
                    continue
                if model.is_polytomous:
                    if item_idx not in category_counts:
                        category_counts[item_idx] = context.expected_category_counts(
                            item_idx,
                            model.n_categories[item_idx],
                            posterior_weights,
                            person_weights,
                        )
                    counts = category_counts[item_idx]
                    item_observed = counts.sum(axis=1)
                    item_correct = None
                else:
                    counts = None
                    item_observed = observed[item_idx]
                    item_correct = correct[item_idx]
                item_se = self._compute_item_se(
                    model,
                    item_idx,
                    name,
                    responses,
                    posterior_weights,
                    r_k=item_correct,
                    n_k_valid=item_observed,
                    r_kc=counts,
                )
                if values.ndim == 1:
                    se[item_idx] = item_se
                else:
                    se[item_idx] = item_se

            se[~free_mask] = 0.0
            standard_errors[name] = se

        return standard_errors

    def _compute_item_se(
        self,
        model: BaseItemModel,
        item_idx: int,
        param_name: str,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        *,
        r_k: NDArray[np.float64] | None = None,
        n_k_valid: NDArray[np.float64] | None = None,
        r_kc: NDArray[np.float64] | None = None,
    ) -> float | NDArray[np.float64]:
        return _compute_item_se_curvature(
            model,
            item_idx,
            param_name,
            responses,
            self._quadrature,
            posterior_weights,
            self.se_step_size,
            scheme="central",
            r_k=r_k,
            n_k_valid=n_k_valid,
            r_kc=r_kc,
            epsilon=self.prob_epsilon,
        )

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
