from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from mirt._backend_config import should_use_rust
from mirt._model_defaults import uses_builtin_model_hooks
from mirt.backends.rust.multigroup import (
    multigroup_e_step_2pl,
    multigroup_e_step_3pl,
    multigroup_e_step_gpcm,
    multigroup_e_step_grm,
    multigroup_e_step_nrm,
)
from mirt.estimation._dichotomous_objective import prepare_dichotomous_objective
from mirt.estimation._em_context import EMFitContext
from mirt.estimation.base import _initialize_free_parameters
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.multigroup._identification import infer_latent_identification
from mirt.multigroup.invariance import InvarianceSpec, parse_invariance
from mirt.multigroup.latent import MultigroupLatentDensity
from mirt.multigroup.results import MultigroupFitResult
from mirt.utils.data import validate_responses
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.multigroup.latent import GroupLatentDistribution
    from mirt.multigroup.model import MultigroupModel

# A pure item objective, its full parameter vector, the free vector positions
# and the optimizer variables placed there.
_BinaryTerm = tuple[
    Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    NDArray[np.float64],
    NDArray[np.intp],
    NDArray[np.intp],
]
# Largest relative L-BFGS-B tolerance for a joint binary item solve. Item
# losses are large sums, so looser values stop on flat 3PL/4PL directions
# long before the item optimum, and EM then crawls toward the maximum.
_JOINT_ITEM_FTOL = 1e-10
_PARAMETER_BOUNDS = {
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
        Relative tolerance for item parameter optimization. Built-in 1PL-4PL
        items, whose free coordinates are solved jointly, use at most 1e-10.
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
        for name, value, minimum in (
            ("n_quadpts", n_quadpts, 5),
            ("max_iter", max_iter, 1),
            ("item_optim_maxiter", item_optim_maxiter, 1),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        for name, value in (("tol", tol), ("item_optim_ftol", item_optim_ftol)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(prob_epsilon) or not 0 < prob_epsilon < 0.5:
            raise ValueError("prob_epsilon must be between 0 and 0.5")

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
        *,
        fixed_parameters: Mapping[str, Mapping[int, float | NDArray[np.float64]]]
        | None = None,
        mean_order: Sequence[int] | None = None,
        initial_latent: Sequence[GroupLatentDistribution] | None = None,
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
        fixed_parameters : mapping, optional
            Stored parameter name -> item index -> fixed scalar or parameter row.
            These values are applied to all groups and retained throughout fitting.
        mean_order : sequence of int, optional
            Permutation of all group indices in nondecreasing population-mean
            order. Supported for unidimensional Gaussian latent distributions.
        initial_latent : sequence of GroupLatentDistribution, optional
            Starting latent distributions, one per group, such as those of a
            previous fit. Only means and covariances that this fit estimates
            are copied. Together with already fitted group models this
            warm-starts nested refits.

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

        responses = [validate_responses(r, n_items=model.n_items) for r in responses]
        for g, r in enumerate(responses):
            category_counts = (
                np.asarray(model.get_group_model(g)._n_categories)
                if model.is_polytomous
                else np.full(model.n_items, 2)
            )
            if np.any(r >= category_counts[None, :]):
                raise ValueError(f"Group {g} contains invalid response categories")
            if not np.any(r >= 0):
                raise ValueError(f"Group {g} must contain observed responses")
        mean_order = self._validate_mean_order(mean_order, model)

        inv_spec = parse_invariance(invariance)

        self._quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=model.n_factors,
        )

        self._latent_density = MultigroupLatentDensity(
            n_groups=model.n_groups,
            n_factors=model.n_factors,
            reference_group=reference_group,
        )
        if fixed_parameters is not None:
            model.fix_item_parameters(fixed_parameters)
        identification = infer_latent_identification(
            model, inv_spec, reference_group, mean_order=mean_order
        )
        for distribution, (estimate_mean, estimate_cov) in zip(
            self._latent_density.distributions, identification, strict=True
        ):
            distribution.estimate_mean = estimate_mean
            distribution.estimate_cov = estimate_cov
        if initial_latent is not None:
            if len(initial_latent) != model.n_groups:
                raise ValueError(
                    "initial_latent must contain one distribution per group"
                )
            for g, start in enumerate(initial_latent):
                current = self._latent_density.distributions[g]
                if current.estimate_mean or current.estimate_cov:
                    self._latent_density.set_group_distribution(
                        g,
                        mean=start.mean if current.estimate_mean else None,
                        cov=start.cov if current.estimate_cov else None,
                    )
        inv_spec.apply_to_model(model)

        for g in range(model.n_groups):
            group_model = model.get_group_model(g)
            if not group_model._is_fitted:
                _initialize_free_parameters(group_model)
        model.synchronize_shared_parameters()

        self._convergence_history = []
        prev_ll = -np.inf
        n_iterations = 0
        converged = False
        state_is_current = False

        for iteration in range(self.max_iter):
            posterior_weights, group_lls = self._e_step(model, responses)
            state_is_current = True

            current_ll = sum(group_lls)
            self._convergence_history.append(current_ll)

            if self.verbose:
                print(f"Iteration {iteration + 1}: LL = {current_ll:.4f}")

            if abs(current_ll - prev_ll) < self.tol:
                converged = True
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                n_iterations = iteration + 1
                break

            prev_ll = current_ll
            n_iterations = iteration + 1

            self._m_step(model, responses, posterior_weights, inv_spec)
            state_is_current = False

            if mean_order is None:
                for g in range(model.n_groups):
                    n_k = posterior_weights[g].sum(axis=0)
                    self._latent_density.update(self._quadrature.nodes, n_k, g)
            else:
                self._update_ordered_latent_density(posterior_weights, mean_order)

        for g in range(model.n_groups):
            model.get_group_model(g)._is_fitted = True

        if state_is_current:
            final_ll = current_ll
        else:
            posterior_weights, group_lls = self._e_step(model, responses)
            final_ll = sum(group_lls)
            self._convergence_history.append(final_ll)
            converged = abs(final_ll - prev_ll) < self.tol

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
            converged=converged,
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
        if should_use_rust() and self._can_use_rust_e_step(model):
            return self._e_step_rust(model, responses)
        return self._e_step_python(model, responses)

    def _can_use_rust_e_step(self, model: MultigroupModel) -> bool:
        """Check if Rust E-step can be used for this model."""
        if model.n_factors != 1:
            return False
        if not all(
            uses_builtin_model_hooks(group_model, likelihood=True)
            for group_model in model.group_models
        ):
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

            log_prior_mass = self._latent_density.log_quadrature_mass(
                quad_points, quad_weights, g
            )
            log_joint = log_likelihoods + log_prior_mass[None, :]
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
        # Item updates never change which coordinates are free.
        masks = [model.effective_free_parameter_masks(g) for g in range(model.n_groups)]
        for item_idx in range(model.n_items):
            self._optimize_item(
                model, item_idx, responses, posterior_weights, quad_points, masks=masks
            )

        model.synchronize_shared_parameters()

    @staticmethod
    def _validate_mean_order(
        mean_order: Sequence[int] | None, model: MultigroupModel
    ) -> tuple[int, ...] | None:
        if mean_order is None:
            return None
        if model.n_factors != 1:
            raise ValueError("ordered group means require a unidimensional model")
        if isinstance(mean_order, (str, bytes)):
            raise ValueError("mean_order must be a permutation of all group indices")
        try:
            order = tuple(mean_order)
        except TypeError as error:
            raise ValueError(
                "mean_order must be a permutation of all group indices"
            ) from error
        if (
            len(order) != model.n_groups
            or any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                for index in order
            )
            or set(order) != set(range(model.n_groups))
        ):
            raise ValueError("mean_order must be a permutation of all group indices")
        return tuple(int(index) for index in order)

    def _update_ordered_latent_density(
        self,
        posterior_weights: list[NDArray[np.float64]],
        order: tuple[int, ...],
    ) -> None:
        """Maximize the discretized Gaussian prior Q under mean inequalities.

        Prior masses include the quadrature reference-density correction and
        their normalizer. Gradients compare posterior and candidate prior
        moments, so the update optimizes the same grid used by the E-step.
        The reference mean and covariance never enter the optimization vector.
        """
        density = self._latent_density
        nodes = self._quadrature.nodes[:, 0]
        log_base = np.log(self._quadrature.weights) + 0.5 * nodes**2
        free = [g for g in range(density.n_groups) if g != density.reference_group]
        n_free = len(free)
        covariance_slots = {
            group: n_free + slot
            for slot, group in enumerate(
                group for group in free if density.distributions[group].estimate_cov
            )
        }
        counts = [weights.sum(axis=0) for weights in posterior_weights]
        x0 = np.array(
            [density.distributions[g].mean[0] for g in free]
            + [np.log(density.distributions[g].cov[0, 0]) for g in covariance_slots]
        )

        def objective(
            params: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            loss = 0.0
            gradient = np.zeros_like(params)
            for slot, group in enumerate(free):
                centered = nodes - params[slot]
                variance = (
                    np.exp(params[covariance_slots[group]])
                    if group in covariance_slots
                    else density.distributions[group].cov[0, 0]
                )
                log_mass = log_base - centered**2 / (2 * variance)
                log_mass -= logsumexp(log_mass)
                mass = np.exp(log_mass)
                n_k = counts[group]
                total = n_k.sum()
                loss -= float(n_k @ log_mass)
                gradient[slot] = -(n_k @ nodes - total * (mass @ nodes)) / variance
                if group in covariance_slots:
                    gradient[covariance_slots[group]] = -(
                        n_k @ (centered**2) - total * (mass @ (centered**2))
                    ) / (2 * variance)
            return loss, gradient

        order_matrix = np.zeros((len(order) - 1, len(x0)))
        for row, (left, right) in enumerate(zip(order[:-1], order[1:], strict=True)):
            if left in free:
                order_matrix[row, free.index(left)] = -1
            if right in free:
                order_matrix[row, free.index(right)] = 1
        result = minimize(
            objective,
            x0=x0,
            jac=True,
            method="SLSQP",
            bounds=[(None, None)] * n_free + [(-14.0, 14.0)] * len(covariance_slots),
            constraints={
                "type": "ineq",
                "fun": lambda params: order_matrix @ params,
                "jac": lambda params: order_matrix,
            },
            options={"maxiter": max(100, self.item_optim_maxiter), "ftol": 1e-9},
        )
        # SLSQP allows small feasibility tolerances. Remove only numerical
        # violations while retaining the reference mean exactly at zero.
        if np.all(np.isfinite(result.x)) and np.min(order_matrix @ result.x) >= -1e-8:
            means = np.zeros(density.n_groups)
            means[free] = result.x[:n_free]
            ordered_means = np.maximum.accumulate(means[list(order)])
            ordered_means -= ordered_means[order.index(density.reference_group)]
            means[list(order)] = ordered_means
            result.x[:n_free] = means[free]
        initial_loss = objective(x0)[0]
        final_loss = objective(result.x)[0]
        if (
            not np.all(np.isfinite(result.x))
            or not np.isfinite(final_loss)
            or np.min(order_matrix @ result.x) < -1e-8
            or final_loss > initial_loss + 1e-7 * max(1.0, abs(initial_loss))
            or not result.success
        ):
            raise RuntimeError(
                f"ordered latent-density optimization failed: {result.message}"
            )
        for slot, group in enumerate(free):
            density.set_group_distribution(
                group,
                mean=np.array([result.x[slot]]),
                cov=(
                    np.array([[np.exp(result.x[covariance_slots[group]])]])
                    if group in covariance_slots
                    else None
                ),
            )

    def _optimize_item(
        self,
        model: MultigroupModel,
        item_idx: int,
        responses: list[NDArray[np.int_]],
        posterior_weights: list[NDArray[np.float64]],
        quad_points: NDArray[np.float64],
        *,
        masks: Sequence[Mapping[str, NDArray[np.bool_]]] | None = None,
    ) -> None:
        """Optimize parameters for a single item across all groups.

        Built-in binary items update every free coordinate in one block (see
        ``_optimize_binary_block``). Other items update one parameter name at
        a time, jointly across groups when it is shared. ``masks`` supplies
        each group's effective free-parameter masks when already computed.
        """
        groups = model.group_models
        counts = [
            self._expected_item_counts(group, item_idx, data, posterior)
            for group, data, posterior in zip(
                groups, responses, posterior_weights, strict=True
            )
        ]
        if masks is None:
            masks = [
                model.effective_free_parameter_masks(g) for g in range(model.n_groups)
            ]
        item_masks = {
            name: [group_masks[name][item_idx].ravel() for group_masks in masks]
            for name in model.parameter_names
        }
        shared = {
            name
            for name in item_masks
            if model.is_item_parameter_shared(name, item_idx)
        }
        if self._optimize_binary_block(
            groups, item_idx, item_masks, shared, counts, quad_points
        ):
            return
        for param_name, name_masks in item_masks.items():
            if param_name in shared:
                self._optimize_parameter_block(
                    groups, name_masks, counts, item_idx, param_name, quad_points
                )
                continue
            for group, mask, group_counts in zip(
                groups, name_masks, counts, strict=True
            ):
                self._optimize_parameter_block(
                    [group], [mask], [group_counts], item_idx, param_name, quad_points
                )

    @staticmethod
    def _expected_item_counts(
        group_model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        posterior: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Reduce category indicators without copying posterior rows."""
        n_categories = (
            group_model._n_categories[item_idx] if group_model.is_polytomous else 2
        )
        return EMFitContext(responses).expected_category_counts(
            item_idx, n_categories, posterior
        )

    def _optimize_shared_item_param(
        self,
        model: MultigroupModel,
        item_idx: int,
        param_name: str,
        responses: list[NDArray[np.int_]],
        posterior_weights: list[NDArray[np.float64]],
        quad_points: NDArray[np.float64],
        *,
        counts: list[NDArray[np.float64]] | None = None,
    ) -> None:
        """Optimize one shared block against every group's expected likelihood."""
        models = model.group_models
        masks = [
            model.effective_free_parameter_masks(g)[param_name][item_idx].ravel()
            for g in range(model.n_groups)
        ]
        if not np.any(masks):
            return
        if counts is None:
            counts = [
                self._expected_item_counts(group, item_idx, data, posterior)
                for group, data, posterior in zip(
                    models, responses, posterior_weights, strict=True
                )
            ]
        if not self._optimize_binary_block(
            models, item_idx, {param_name: masks}, {param_name}, counts, quad_points
        ):
            self._optimize_parameter_block(
                models, masks, counts, item_idx, param_name, quad_points
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
        *,
        counts: NDArray[np.float64] | None = None,
    ) -> None:
        """Optimize only a group's structurally free, unfixed coordinates."""
        group = model.get_group_model(group_idx)
        mask = model.effective_free_parameter_masks(group_idx)[param_name][
            item_idx
        ].ravel()
        if not np.any(mask):
            return
        if counts is None:
            counts = self._expected_item_counts(
                group, item_idx, group_responses, group_weights
            )
        if not self._optimize_binary_block(
            [group], item_idx, {param_name: [mask]}, (), [counts], quad_points
        ):
            self._optimize_parameter_block(
                [group], [mask], [counts], item_idx, param_name, quad_points
            )

    def _optimize_binary_block(
        self,
        groups: Sequence[BaseItemModel],
        item_idx: int,
        masks: Mapping[str, Sequence[NDArray[np.bool_]]],
        shared: Collection[str],
        counts: Sequence[NDArray[np.float64]],
        quad_points: NDArray[np.float64],
    ) -> bool:
        """Maximize a built-in binary item's free coordinates with one solver.

        ``masks`` maps parameter names to each group's free coordinates of the
        item; omitted names stay fixed. A free coordinate of a ``shared`` name
        is one variable for every group that frees it, and other coordinates
        are one variable per group. Without shared variables each group is
        solved on its own, and groups whose item curves and variables
        coincide pool their counts. A solve is kept only when it improves the
        expected log-likelihood, as generalized EM requires.

        Returns
        -------
        bool
            ``False``, with every model unchanged, when the item needs the
            per-parameter path: polytomous, custom, or restricted models.
        """
        first = groups[0]
        # Pooled groups share one kernel, so every group needs the built-in
        # curve, not only the group whose kernel is prepared.
        if first.is_polytomous or any(
            type(group) is not type(first) or not uses_builtin_model_hooks(group)
            for group in groups
        ):
            return False
        fixed_slope = first.model_name == "1PL"
        if fixed_slope and np.any(masks.get("discrimination", False)):
            return False
        parameters = [group.parameters for group in groups]
        names = [
            name
            for name in parameters[0]
            if not (fixed_slope and name == "discrimination")
        ]
        rows = [
            np.concatenate([np.ravel(values[name][item_idx]) for name in names])
            for values in parameters
        ]
        observed = [bool(np.any(group_counts)) for group_counts in counts]
        lower: list[float] = []
        upper: list[float] = []
        spans: dict[str, slice] = {}
        positions: list[list[int]] = [[] for _ in groups]
        variables: list[list[int]] = [[] for _ in groups]
        start: list[float] = []
        box: list[tuple[float, float]] = []
        coupled = False
        for name in names:
            offset = len(lower)
            size = np.size(parameters[0][name][item_idx])
            spans[name] = slice(offset, offset + size)
            bound = self._parameter_bound(first, name)
            lower.extend([bound[0]] * size)
            upper.extend([bound[1]] * size)
            name_masks = masks.get(name)
            if name_masks is None:
                continue
            if name in shared:
                for coordinate in np.flatnonzero(np.logical_or.reduce(name_masks)):
                    coupled = True
                    for g, mask in enumerate(name_masks):
                        if mask[coordinate]:
                            positions[g].append(offset + coordinate)
                            variables[g].append(len(start))
                            value = rows[g][offset + coordinate]
                    start.append(value)
                    box.append(bound)
                continue
            for g, mask in enumerate(name_masks):
                if not observed[g]:
                    continue
                for coordinate in np.flatnonzero(mask):
                    positions[g].append(offset + coordinate)
                    variables[g].append(len(start))
                    start.append(rows[g][offset + coordinate])
                    box.append(bound)

        members = [g for g in range(len(groups)) if variables[g]]
        problems = [members] if coupled else [[g] for g in members]
        initial = np.asarray(start, dtype=np.float64)
        solves = []
        for problem in problems:
            terms_groups = [g for g in problem if observed[g]]
            if not terms_groups:
                continue
            used = np.unique(np.concatenate([variables[g] for g in problem]))
            local = {g: np.searchsorted(used, variables[g]) for g in problem}
            reference = terms_groups[0]
            pooled = all(
                positions[g] == positions[reference]
                and variables[g] == variables[reference]
                and all(
                    np.array_equal(values[item_idx], parameters[g][name][item_idx])
                    for name, values in parameters[reference].items()
                )
                for g in terms_groups[1:]
            )
            terms: list[_BinaryTerm] = []
            for term in [terms_groups] if pooled else [[g] for g in terms_groups]:
                g = term[0]
                index = np.asarray(positions[g], dtype=np.intp)
                trial = rows[g].copy()
                trial[index] = initial[variables[g]]
                # The kernel skips overflow guards when every evaluated vector
                # lies in its box, so cover fixed values and the start too.
                kernel_box = list(
                    zip(
                        np.minimum(lower, np.minimum(rows[g], trial)),
                        np.maximum(upper, np.maximum(rows[g], trial)),
                        strict=True,
                    )
                )
                term_counts = np.sum([counts[h] for h in term], axis=0)
                kernel = prepare_dichotomous_objective(
                    groups[g],
                    item_idx,
                    quad_points,
                    term_counts.sum(axis=1),
                    term_counts[:, 1],
                    self.prob_epsilon,
                    kernel_box,
                )
                if kernel is None:
                    return False
                terms.append((kernel, rows[g], index, local[g]))
            solves.append((problem, used, local, terms))

        for problem, used, local, terms in solves:
            solution = self._minimize_binary_terms(
                terms, initial[used], [box[index] for index in used]
            )
            if solution is None:
                continue
            for g in problem:
                row = rows[g].copy()
                row[positions[g]] = solution[local[g]]
                for name, span in spans.items():
                    if not np.array_equal(row[span], rows[g][span]):
                        self._set_param(groups[g], item_idx, name, row[span])
        return True

    def _minimize_binary_terms(
        self,
        terms: list[_BinaryTerm],
        start: NDArray[np.float64],
        box: list[tuple[float, float]],
    ) -> NDArray[np.float64] | None:
        """Minimize summed kernel losses; keep only an improving solution."""

        def objective(
            params: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            loss = 0.0
            gradient = np.zeros_like(params)
            for kernel, base, index, variables in terms:
                trial = base.copy()
                trial[index] = params[variables]
                value, full_gradient = kernel(trial)
                loss += value
                gradient[variables] += full_gradient[index]
            return loss, gradient

        initial_loss = objective(start)[0]
        result = minimize(
            objective,
            x0=start,
            jac=True,
            method="L-BFGS-B",
            bounds=box,
            options={
                "maxiter": self.item_optim_maxiter,
                "ftol": min(self.item_optim_ftol, _JOINT_ITEM_FTOL),
            },
        )
        if not np.all(np.isfinite(result.x)):
            return None
        final_loss = objective(result.x)[0]
        if not np.isfinite(final_loss) or final_loss > initial_loss + 1e-10 * max(
            1.0, abs(initial_loss)
        ):
            return None
        return np.asarray(result.x, dtype=np.float64)

    def _optimize_parameter_block(
        self,
        models: list[BaseItemModel],
        masks: list[NDArray[np.bool_]],
        counts: list[NDArray[np.float64]],
        item_idx: int,
        param_name: str,
        quad_points: NDArray[np.float64],
    ) -> None:
        """Maximize a free block and restore state after rejected trials.

        Trials go through the models' public probability and setter hooks.
        An incomplete optimizer step is accepted only when it improves the
        actual expected likelihood, as required by generalized EM.
        """
        if not np.any(masks) or not any(
            np.any(group_counts) for group_counts in counts
        ):
            return
        originals = [
            group.parameters[param_name][item_idx].ravel().copy() for group in models
        ]
        active = np.logical_or.reduce(masks)
        current, bounds = self._get_param_and_bounds(models[0], item_idx, param_name)
        for original, mask in zip(originals, masks, strict=True):
            current[mask] = original[mask]
        slots = np.flatnonzero(active)
        evaluations = list(zip(models, originals, masks, counts, strict=True))
        if len(models) > 1 and all(
            type(group) is type(models[0])
            and uses_builtin_model_hooks(group)
            and np.array_equal(mask, masks[0])
            and all(
                np.array_equal(group.parameters[name][item_idx], value[item_idx])
                for name, value in models[0].parameters.items()
            )
            for group, mask in zip(models, masks, strict=True)
        ):
            # Identical curves share sufficient statistics; retain separate
            # likelihood terms whenever another group parameter or hook differs.
            evaluations = [(models[0], originals[0], masks[0], np.sum(counts, axis=0))]

        def evaluate(params: NDArray[np.float64]) -> float:
            loss = 0.0
            for group, original, mask, group_counts in evaluations:
                row = original.copy()
                row[slots[mask[slots]]] = params[mask[slots]]
                self._set_param(group, item_idx, param_name, row)
                probs = group.probability(quad_points, item_idx)
                if not group.is_polytomous:
                    probs = np.column_stack((1 - probs, probs))
                if not np.all(np.isfinite(probs)):
                    return np.inf
                loss -= float(
                    np.sum(
                        group_counts
                        * np.log(
                            np.clip(probs, self.prob_epsilon, 1 - self.prob_epsilon)
                        )
                    )
                )
            return loss

        accepted = False
        try:
            initial_loss = evaluate(current[active])
            ordered_thresholds = param_name == "thresholds" and all(
                group.model_name == "GRM" for group in models
            )
            constraint_matrix = []
            constraint_constants = []
            if ordered_thresholds:
                for group, original, mask in zip(models, originals, masks, strict=True):
                    for first in range(group._n_categories[item_idx] - 2):
                        row = np.zeros(slots.size)
                        constant = original[first + 1] - original[first]
                        for index, sign in ((first, -1), (first + 1, 1)):
                            if mask[index]:
                                column = int(np.flatnonzero(slots == index)[0])
                                row[column] += sign
                                constant -= sign * original[index]
                        if np.any(row):
                            constant -= 1e-7
                        constraint_matrix.append(row)
                        constraint_constants.append(constant)
                ordered_thresholds = bool(constraint_matrix)
            matrix = np.asarray(constraint_matrix)
            constants = np.asarray(constraint_constants)
            result = minimize(
                evaluate,
                x0=current[active],
                method="SLSQP" if ordered_thresholds else "L-BFGS-B",
                bounds=[bounds[index] for index in slots],
                constraints=(
                    {
                        "type": "ineq",
                        "fun": lambda params: matrix @ params + constants,
                        "jac": lambda params: matrix,
                    }
                    if ordered_thresholds
                    else ()
                ),
                options={
                    "maxiter": self.item_optim_maxiter,
                    "ftol": self.item_optim_ftol,
                },
            )
            if not np.all(np.isfinite(result.x)):
                return
            if ordered_thresholds:
                if np.min(matrix @ result.x + constants) < -1e-9:
                    return
            final_loss = evaluate(result.x)
            if not np.isfinite(final_loss) or final_loss > initial_loss + 1e-10 * max(
                1.0, abs(initial_loss)
            ):
                return
            for group, original, mask in zip(models, originals, masks, strict=True):
                row = original.copy()
                row[slots[mask[slots]]] = result.x[mask[slots]]
                self._set_param(group, item_idx, param_name, row)
            accepted = True
        finally:
            if not accepted:
                for group, original in zip(models, originals, strict=True):
                    self._set_param(group, item_idx, param_name, original)

    @staticmethod
    def _parameter_bound(model: BaseItemModel, param_name: str) -> tuple[float, float]:
        """Return the optimizer box for every coordinate of a parameter."""
        if param_name == "slopes" and model.model_name == "NRM":
            return (-5.0, 5.0)
        if (
            param_name == "discrimination"
            and model.n_factors > 1
            and not model.is_polytomous
        ):
            return (-5.0, 5.0)
        return _PARAMETER_BOUNDS.get(param_name, (-10.0, 10.0))

    def _get_param_and_bounds(
        self,
        model: BaseItemModel,
        item_idx: int,
        param_name: str,
    ) -> tuple[NDArray[np.float64], list[tuple[float, float]]]:
        """Get current parameter value and bounds for optimization."""
        current = np.ravel(model.parameters[param_name][item_idx]).copy()
        return current, [self._parameter_bound(model, param_name)] * current.size

    def _set_param(
        self,
        model: BaseItemModel,
        item_idx: int,
        param_name: str,
        value: NDArray[np.float64],
    ) -> None:
        """Set parameter value for a specific item."""
        values = model._parameters[param_name].copy()
        row_shape = values[item_idx].shape
        values[item_idx] = np.asarray(value, dtype=np.float64).reshape(row_shape)
        canonical = model._canonical_parameter_values(param_name, values)
        row = np.asarray(canonical[item_idx])
        model.set_item_parameter(
            item_idx, param_name, float(row) if row.ndim == 0 else row
        )

    @property
    def convergence_history(self) -> list[float]:
        """Log-likelihood history across iterations."""
        return self._convergence_history.copy()
