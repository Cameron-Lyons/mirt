from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from mirt._rust_backend import (
    RUST_AVAILABLE,
    multigroup_e_step_2pl,
    multigroup_e_step_3pl,
    multigroup_e_step_gpcm,
    multigroup_e_step_grm,
    multigroup_e_step_nrm,
)
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.multigroup.invariance import InvarianceSpec, parse_invariance
from mirt.multigroup.latent import MultigroupLatentDensity
from mirt.multigroup.results import MultigroupFitResult
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.multigroup.model import MultigroupModel


class MultigroupEMEstimator:
    """EM estimator for simultaneous multigroup IRT estimation.

    This estimator fits IRT models across multiple groups simultaneously,
    with support for various invariance constraints and group-specific
    latent distributions.

    Parameters
    ----------
    n_quadpts : int
        Number of quadrature points for numerical integration.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance for log-likelihood change.
    verbose : bool
        Print iteration progress.
    prob_epsilon : float
        Minimum probability for numerical stability.
    item_optim_maxiter : int
        Maximum iterations for item parameter optimization.
    item_optim_ftol : float
        Tolerance for item parameter optimization.
    """

    def __init__(
        self,
        n_quadpts: int = 21,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        prob_epsilon: float = 1e-10,
        item_optim_maxiter: int = 50,
        item_optim_ftol: float = 1e-6,
    ) -> None:
        if n_quadpts < 5:
            raise ValueError("n_quadpts must be at least 5")

        self.n_quadpts = n_quadpts
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.prob_epsilon = prob_epsilon
        self.item_optim_maxiter = item_optim_maxiter
        self.item_optim_ftol = item_optim_ftol

        self._quadrature: GaussHermiteQuadrature | None = None
        self._latent_density: MultigroupLatentDensity | None = None
        self._convergence_history: list[float] = []

    def fit(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        invariance: InvarianceSpec | str = "configural",
        reference_group: int = 0,
    ) -> MultigroupFitResult:
        """Fit multigroup model with simultaneous EM.

        Parameters
        ----------
        model : MultigroupModel
            The multigroup model to fit.
        responses : list of ndarray
            Response matrices, one per group. Each has shape (n_persons_g, n_items).
        invariance : InvarianceSpec or str
            Invariance level or custom specification.
        reference_group : int
            Index of reference group (fixed mean=0, cov=I).

        Returns
        -------
        MultigroupFitResult
            Fitted model results.
        """
        if len(responses) != model.n_groups:
            raise ValueError(
                f"Number of response matrices ({len(responses)}) must match "
                f"n_groups ({model.n_groups})"
            )

        responses = [np.asarray(r) for r in responses]
        for g, r in enumerate(responses):
            if r.shape[1] != model.n_items:
                raise ValueError(
                    f"Group {g} has {r.shape[1]} items, expected {model.n_items}"
                )

        inv_spec = parse_invariance(invariance)
        inv_spec.apply_to_model(model)

        self._quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=model.n_factors,
        )

        self._latent_density = MultigroupLatentDensity(
            n_groups=model.n_groups,
            n_factors=model.n_factors,
            reference_group=reference_group,
        )

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            if not group_model._is_fitted:
                group_model._initialize_parameters()

        self._convergence_history = []
        prev_ll = -np.inf
        n_iterations = 0

        for iteration in range(self.max_iter):
            posterior_weights, group_lls = self._e_step(model, responses)

            current_ll = sum(group_lls)
            self._convergence_history.append(current_ll)

            if self.verbose:
                print(f"Iteration {iteration + 1}: LL = {current_ll:.4f}")

            if abs(current_ll - prev_ll) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                n_iterations = iteration + 1
                break

            prev_ll = current_ll
            n_iterations = iteration + 1

            self._m_step(model, responses, posterior_weights, inv_spec)

            for g in range(model.n_groups):
                n_k = posterior_weights[g].sum(axis=0)
                self._latent_density.update(self._quadrature.nodes, n_k, g)

        for g in range(model.n_groups):
            model.get_group_model(g)._is_fitted = True

        posterior_weights, group_lls = self._e_step(model, responses)
        final_ll = sum(group_lls)

        group_n = [r.shape[0] for r in responses]
        total_n = sum(group_n)

        n_item_params = model.n_parameters
        n_latent_params = self._latent_density.n_parameters
        n_params = n_item_params + n_latent_params

        aic = -2 * final_ll + 2 * n_params
        bic = -2 * final_ll + np.log(total_n) * n_params

        return MultigroupFitResult(
            model=model,
            invariance=inv_spec.level,
            log_likelihood=final_ll,
            n_iterations=n_iterations,
            converged=n_iterations < self.max_iter,
            group_log_likelihoods=group_lls,
            group_n_observations=group_n,
            latent_distributions=[d.copy() for d in self._latent_density.distributions],
            aic=aic,
            bic=bic,
            n_parameters=n_params,
            n_observations=total_n,
        )

    def _e_step(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """E-step: compute posterior weights for each group.

        Returns
        -------
        posterior_weights : list of ndarray
            Posterior weights per group, shape (n_persons_g, n_quad).
        group_lls : list of float
            Marginal log-likelihood per group.
        """
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights
        n_quad = len(quad_weights)

        if RUST_AVAILABLE and self._can_use_rust_e_step(model):
            return self._e_step_rust(model, responses)

        posterior_weights = []
        group_lls = []

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            group_responses = responses[g]
            n_persons = group_responses.shape[0]

            if hasattr(group_model, "log_likelihood_batch"):
                log_likelihoods = group_model.log_likelihood_batch(
                    group_responses, quad_points
                )
            else:
                log_likelihoods = np.zeros((n_persons, n_quad))
                for q in range(n_quad):
                    theta_q = quad_points[q : q + 1]
                    log_likelihoods[:, q] = group_model.log_likelihood(
                        group_responses, theta_q
                    )

            log_prior = self._latent_density.log_density(quad_points, g)

            log_joint = (
                log_likelihoods + log_prior[None, :] + np.log(quad_weights)[None, :]
            )

            log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
            log_posterior = log_joint - log_marginal

            post_w = np.exp(log_posterior)
            posterior_weights.append(post_w)

            group_ll = np.sum(log_marginal)
            group_lls.append(group_ll)

        return posterior_weights, group_lls

    def _can_use_rust_e_step(self, model: MultigroupModel) -> bool:
        """Check if Rust E-step can be used for this model."""
        if model.n_factors != 1:
            return False
        model_name = model.model_name
        if model.is_polytomous:
            return model_name in ("GRM", "GPCM", "PCM", "NRM")
        return model_name in ("2PL", "1PL", "3PL")

    def _e_step_rust(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """E-step using Rust backend for parallel processing."""
        quad_points = self._quadrature.nodes.ravel()
        quad_weights = self._quadrature.weights

        prior_means = np.array(
            [
                self._latent_density.distributions[g].mean[0]
                for g in range(model.n_groups)
            ]
        )
        prior_vars = np.array(
            [
                self._latent_density.distributions[g].cov[0, 0]
                for g in range(model.n_groups)
            ]
        )

        if model.model_name == "GRM":
            return self._e_step_rust_grm(
                model, responses, quad_points, quad_weights, prior_means, prior_vars
            )

        if model.model_name in ("GPCM", "PCM"):
            return self._e_step_rust_gpcm(
                model, responses, quad_points, quad_weights, prior_means, prior_vars
            )

        if model.model_name == "NRM":
            return self._e_step_rust_nrm(
                model, responses, quad_points, quad_weights, prior_means, prior_vars
            )

        disc_list = []
        diff_list = []
        guess_list = []

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            params = group_model.parameters

            disc = params.get("discrimination", np.ones(model.n_items))
            diff = params.get("difficulty", np.zeros(model.n_items))

            disc_list.append(disc.ravel())
            diff_list.append(diff.ravel())

            if "guessing" in params:
                guess_list.append(params["guessing"].ravel())

        if model.model_name in ("2PL", "1PL"):
            result = multigroup_e_step_2pl(
                responses,
                quad_points,
                quad_weights,
                disc_list,
                diff_list,
                prior_means,
                prior_vars,
            )
        else:
            result = multigroup_e_step_3pl(
                responses,
                quad_points,
                quad_weights,
                disc_list,
                diff_list,
                guess_list,
                prior_means,
                prior_vars,
            )

        if result is None:
            return self._e_step_python(model, responses)

        posterior_weights, group_lls = result
        return list(posterior_weights), list(group_lls)

    def _e_step_rust_grm(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        quad_points: NDArray[np.float64],
        quad_weights: NDArray[np.float64],
        prior_means: NDArray[np.float64],
        prior_vars: NDArray[np.float64],
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """E-step using Rust backend for GRM models."""
        disc_list = []
        thresh_list = []
        n_categories_list = []

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            params = group_model.parameters

            disc = params.get("discrimination", np.ones(model.n_items))
            thresh = params.get("thresholds", np.zeros((model.n_items, 1)))
            n_cats = np.array(group_model._n_categories, dtype=np.int32)

            disc_list.append(disc.ravel())
            thresh_list.append(thresh)
            n_categories_list.append(n_cats)

        result = multigroup_e_step_grm(
            responses,
            quad_points,
            quad_weights,
            disc_list,
            thresh_list,
            n_categories_list,
            prior_means,
            prior_vars,
        )

        if result is None:
            return self._e_step_python(model, responses)

        posterior_weights, group_lls = result
        return list(posterior_weights), list(group_lls)

    def _e_step_rust_gpcm(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        quad_points: NDArray[np.float64],
        quad_weights: NDArray[np.float64],
        prior_means: NDArray[np.float64],
        prior_vars: NDArray[np.float64],
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """E-step using Rust backend for GPCM models."""
        disc_list = []
        steps_list = []
        n_categories_list = []

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            params = group_model.parameters

            disc = params.get("discrimination", np.ones(model.n_items))
            steps = params.get("steps", np.zeros((model.n_items, 1)))
            n_cats = np.array(group_model._n_categories, dtype=np.int32)

            max_cats = max(group_model._n_categories)
            steps_full = np.zeros((model.n_items, max_cats))
            for j, nc in enumerate(group_model._n_categories):
                steps_full[j, 1:nc] = steps[j, : nc - 1]

            disc_list.append(disc.ravel())
            steps_list.append(steps_full)
            n_categories_list.append(n_cats)

        result = multigroup_e_step_gpcm(
            responses,
            quad_points,
            quad_weights,
            disc_list,
            steps_list,
            n_categories_list,
            prior_means,
            prior_vars,
        )

        if result is None:
            return self._e_step_python(model, responses)

        posterior_weights, group_lls = result
        return list(posterior_weights), list(group_lls)

    def _e_step_rust_nrm(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        quad_points: NDArray[np.float64],
        quad_weights: NDArray[np.float64],
        prior_means: NDArray[np.float64],
        prior_vars: NDArray[np.float64],
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """E-step using Rust backend for NRM models."""
        slopes_list = []
        intercepts_list = []
        n_categories_list = []

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            params = group_model.parameters

            slopes = params.get("slopes", np.zeros((model.n_items, 1)))
            intercepts = params.get("intercepts", np.zeros((model.n_items, 1)))
            n_cats = np.array(group_model._n_categories, dtype=np.int32)

            max_cats = max(group_model._n_categories)
            slopes_full = np.zeros((model.n_items, max_cats))
            intercepts_full = np.zeros((model.n_items, max_cats))
            for j, nc in enumerate(group_model._n_categories):
                slopes_full[j, :nc] = slopes[j, :nc]
                intercepts_full[j, :nc] = intercepts[j, :nc]

            slopes_list.append(slopes_full)
            intercepts_list.append(intercepts_full)
            n_categories_list.append(n_cats)

        result = multigroup_e_step_nrm(
            responses,
            quad_points,
            quad_weights,
            slopes_list,
            intercepts_list,
            n_categories_list,
            prior_means,
            prior_vars,
        )

        if result is None:
            return self._e_step_python(model, responses)

        posterior_weights, group_lls = result
        return list(posterior_weights), list(group_lls)

    def _e_step_python(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
    ) -> tuple[list[NDArray[np.float64]], list[float]]:
        """Fallback Python E-step implementation."""
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights
        n_quad = len(quad_weights)

        posterior_weights = []
        group_lls = []

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            group_responses = responses[g]
            n_persons = group_responses.shape[0]

            if hasattr(group_model, "log_likelihood_batch"):
                log_likelihoods = group_model.log_likelihood_batch(
                    group_responses, quad_points
                )
            else:
                log_likelihoods = np.zeros((n_persons, n_quad))
                for q in range(n_quad):
                    theta_q = quad_points[q : q + 1]
                    log_likelihoods[:, q] = group_model.log_likelihood(
                        group_responses, theta_q
                    )

            log_prior = self._latent_density.log_density(quad_points, g)
            log_joint = (
                log_likelihoods + log_prior[None, :] + np.log(quad_weights)[None, :]
            )
            log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
            log_posterior = log_joint - log_marginal

            post_w = np.exp(log_posterior)
            posterior_weights.append(post_w)
            group_ll = np.sum(log_marginal)
            group_lls.append(group_ll)

        return posterior_weights, group_lls

    def _m_step(
        self,
        model: MultigroupModel,
        responses: list[NDArray[np.int_]],
        posterior_weights: list[NDArray[np.float64]],
        invariance: InvarianceSpec,
    ) -> None:
        """M-step: update parameters respecting constraints.

        For shared parameters: aggregate expected sufficient statistics
        across groups and optimize once.
        For group-specific parameters: optimize independently per group.
        """
        quad_points = self._quadrature.nodes
        n_items = model.n_items

        for item_idx in range(n_items):
            self._optimize_item(
                model, item_idx, responses, posterior_weights, quad_points
            )

        model.synchronize_shared_parameters()

    def _optimize_item(
        self,
        model: MultigroupModel,
        item_idx: int,
        responses: list[NDArray[np.int_]],
        posterior_weights: list[NDArray[np.float64]],
        quad_points: NDArray[np.float64],
    ) -> None:
        """Optimize parameters for a single item across all groups."""
        for param_name in model.parameter_names:
            if model.is_item_parameter_shared(param_name, item_idx):
                self._optimize_shared_item_param(
                    model,
                    item_idx,
                    param_name,
                    responses,
                    posterior_weights,
                    quad_points,
                )
            else:
                for g in range(model.n_groups):
                    self._optimize_group_item_param(
                        model,
                        g,
                        item_idx,
                        param_name,
                        responses[g],
                        posterior_weights[g],
                        quad_points,
                    )

    def _optimize_shared_item_param(
        self,
        model: MultigroupModel,
        item_idx: int,
        param_name: str,
        responses: list[NDArray[np.int_]],
        posterior_weights: list[NDArray[np.float64]],
        quad_points: NDArray[np.float64],
    ) -> None:
        """Optimize a shared parameter by aggregating across groups."""
        group_model = model.get_group_model(0)
        current_params, bounds = self._get_param_and_bounds(
            group_model, item_idx, param_name
        )

        eps = self.prob_epsilon

        if model.is_polytomous:
            n_categories = group_model._n_categories[item_idx]
            n_quad = quad_points.shape[0]

            aggregated_r_kc = np.zeros((n_quad, n_categories))
            aggregated_n_k = np.zeros(n_quad)

            for g in range(model.n_groups):
                item_responses = responses[g][:, item_idx]
                valid_mask = item_responses >= 0
                group_weights = posterior_weights[g]

                aggregated_n_k += np.sum(group_weights[valid_mask], axis=0)

                for c in range(n_categories):
                    cat_mask = valid_mask & (item_responses == c)
                    aggregated_r_kc[:, c] += np.sum(group_weights[cat_mask, :], axis=0)

            def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
                self._set_param(group_model, item_idx, param_name, params)

                probs = group_model.probability(quad_points, item_idx)
                probs = np.clip(probs, eps, 1 - eps)

                ll = np.sum(aggregated_r_kc * np.log(probs))
                return -ll

        else:
            aggregated_r_k = np.zeros(quad_points.shape[0])
            aggregated_n_k = np.zeros(quad_points.shape[0])

            for g in range(model.n_groups):
                item_responses = responses[g][:, item_idx]
                valid_mask = item_responses >= 0
                group_weights = posterior_weights[g]

                aggregated_n_k += np.sum(group_weights[valid_mask], axis=0)
                aggregated_r_k += np.sum(
                    item_responses[valid_mask, None] * group_weights[valid_mask, :],
                    axis=0,
                )

            def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
                self._set_param(group_model, item_idx, param_name, params)

                probs = group_model.probability(quad_points, item_idx)
                probs = np.clip(probs, eps, 1 - eps)

                ll = np.sum(
                    aggregated_r_k * np.log(probs)
                    + (aggregated_n_k - aggregated_r_k) * np.log(1 - probs)
                )
                return -ll

        result = minimize(
            neg_expected_log_likelihood,
            x0=current_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.item_optim_maxiter, "ftol": self.item_optim_ftol},
        )

        optimized_value = result.x
        for g in range(model.n_groups):
            self._set_param(
                model.get_group_model(g), item_idx, param_name, optimized_value
            )

    def _optimize_group_item_param(
        self,
        model: MultigroupModel,
        group_idx: int,
        item_idx: int,
        param_name: str,
        group_responses: NDArray[np.int_],
        group_weights: NDArray[np.float64],
        quad_points: NDArray[np.float64],
    ) -> None:
        """Optimize a group-specific parameter."""
        group_model = model.get_group_model(group_idx)
        current_params, bounds = self._get_param_and_bounds(
            group_model, item_idx, param_name
        )

        item_responses = group_responses[:, item_idx]
        valid_mask = item_responses >= 0

        n_k_valid = np.sum(group_weights[valid_mask], axis=0)

        eps = self.prob_epsilon

        if model.is_polytomous:
            n_categories = group_model._n_categories[item_idx]
            n_quad = quad_points.shape[0]

            r_kc = np.zeros((n_quad, n_categories))
            for c in range(n_categories):
                cat_mask = valid_mask & (item_responses == c)
                r_kc[:, c] = np.sum(group_weights[cat_mask, :], axis=0)

            def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
                self._set_param(group_model, item_idx, param_name, params)

                probs = group_model.probability(quad_points, item_idx)
                probs = np.clip(probs, eps, 1 - eps)

                ll = np.sum(r_kc * np.log(probs))
                return -ll

        else:
            r_k = np.sum(
                item_responses[valid_mask, None] * group_weights[valid_mask, :],
                axis=0,
            )

            def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
                self._set_param(group_model, item_idx, param_name, params)

                probs = group_model.probability(quad_points, item_idx)
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

        self._set_param(group_model, item_idx, param_name, result.x)

    def _get_param_and_bounds(
        self,
        model: BaseItemModel,
        item_idx: int,
        param_name: str,
    ) -> tuple[NDArray[np.float64], list[tuple[float, float]]]:
        """Get current parameter value and bounds for optimization."""
        values = model.parameters[param_name]

        if values.ndim == 1:
            current = np.array([values[item_idx]])
        else:
            current = values[item_idx].copy()

        bounds_map = {
            "discrimination": (0.1, 5.0),
            "slopes": (0.1, 5.0),
            "loadings": (-5.0, 5.0),
            "general_loadings": (0.1, 5.0),
            "specific_loadings": (0.1, 5.0),
            "difficulty": (-6.0, 6.0),
            "intercepts": (-6.0, 6.0),
            "location": (-6.0, 6.0),
            "thresholds": (-6.0, 6.0),
            "steps": (-6.0, 6.0),
            "guessing": (0.0, 0.5),
            "slipping": (0.5, 1.0),
            "upper": (0.5, 1.0),
        }

        default_bounds = (-10.0, 10.0)
        bound = bounds_map.get(param_name, default_bounds)
        bounds = [bound] * len(current)

        return current, bounds

    def _set_param(
        self,
        model: BaseItemModel,
        item_idx: int,
        param_name: str,
        value: NDArray[np.float64],
    ) -> None:
        """Set parameter value for a specific item."""
        values = model._parameters[param_name]

        if values.ndim == 1:
            values[item_idx] = float(value[0]) if len(value) == 1 else float(value)
        else:
            values[item_idx] = value

    @property
    def convergence_history(self) -> list[float]:
        """Log-likelihood history across iterations."""
        return self._convergence_history.copy()
