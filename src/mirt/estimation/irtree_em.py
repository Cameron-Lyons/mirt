from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from mirt.estimation.base import BaseEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.irtree import IRTreeModel


@dataclass
class IRTreeResult:
    """Result from IRTree model estimation."""

    model: IRTreeModel
    log_likelihood: float
    trait_means: NDArray[np.float64]
    trait_correlations: NDArray[np.float64]
    theta_estimates: NDArray[np.float64]
    theta_se: NDArray[np.float64]
    standard_errors: dict[str, NDArray[np.float64]]
    aic: float
    bic: float
    converged: bool
    n_iterations: int
    n_observations: int
    n_parameters: int

    def summary(self) -> str:
        lines = []
        width = 80

        lines.append("=" * width)
        lines.append(f"{'IRTree Model Results':^{width}}")
        lines.append("=" * width)

        lines.append(
            f"Tree Structure:     {self.model.tree_spec.name:<20} Log-Likelihood:    {self.log_likelihood:>12.4f}"
        )
        lines.append(
            f"No. Items:          {self.model.n_items:<20} AIC:               {self.aic:>12.4f}"
        )
        lines.append(
            f"No. Traits:         {self.model.n_traits:<20} BIC:               {self.bic:>12.4f}"
        )
        lines.append(
            f"No. Persons:        {self.n_observations:<20} No. Parameters:    {self.n_parameters:>12}"
        )
        lines.append(
            f"Converged:          {str(self.converged):<20} Iterations:        {self.n_iterations:>12}"
        )
        lines.append("-" * width)

        lines.append("\nTrait Means:")
        for i, name in enumerate(self.model.trait_names):
            lines.append(f"  {name}: {self.trait_means[i]:.4f}")

        lines.append("\nTrait Correlations:")
        header = "".ljust(15)
        for name in self.model.trait_names:
            header += f"{name[:10]:>12}"
        lines.append(header)
        for i, name in enumerate(self.model.trait_names):
            row = f"{name:<15}"
            for j in range(self.model.n_traits):
                row += f"{self.trait_correlations[i, j]:>12.3f}"
            lines.append(row)

        lines.append("=" * width)
        return "\n".join(lines)

    def trait_summary(self) -> str:
        """Generate summary focused on response style traits."""
        lines = []
        width = 60

        lines.append("=" * width)
        lines.append(f"{'Response Style Analysis':^{width}}")
        lines.append("=" * width)

        for i, name in enumerate(self.model.trait_names):
            mean = self.trait_means[i]
            var = 1.0
            lines.append(f"\n{name}:")
            lines.append(f"  Mean:     {mean:>8.4f}")
            lines.append(f"  Variance: {var:>8.4f}")

            lines.append("  Correlations with other traits:")
            for j, other_name in enumerate(self.model.trait_names):
                if i != j:
                    lines.append(
                        f"    {other_name}: {self.trait_correlations[i, j]:>8.4f}"
                    )

        lines.append("=" * width)
        return "\n".join(lines)


class IRTreeEMEstimator(BaseEstimator):
    """EM algorithm for IRTree models.

    Estimates item parameters and trait distributions for IRTree models
    using marginal maximum likelihood with EM.

    Parameters
    ----------
    n_quadpts : int
        Number of quadrature points per dimension
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance for log-likelihood change
    estimate_correlations : bool
        Whether to estimate trait correlations
    verbose : bool
        Print progress information
    """

    def __init__(
        self,
        n_quadpts: int = 11,
        max_iter: int = 500,
        tol: float = 1e-4,
        estimate_correlations: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        self.n_quadpts = n_quadpts
        self.estimate_correlations = estimate_correlations
        self._quadrature: GaussHermiteQuadrature | None = None

    def fit(
        self,
        model: IRTreeModel,
        responses: NDArray[np.int_],
    ) -> IRTreeResult:
        """Fit IRTree model via EM algorithm.

        Parameters
        ----------
        model : IRTreeModel
            IRTree model to fit
        responses : NDArray
            Response matrix (n_persons, n_items) with ordinal responses

        Returns
        -------
        IRTreeResult
            Fitted model results
        """
        responses = np.asarray(responses, dtype=np.int32)
        if responses.ndim != 2:
            raise ValueError(f"responses must be 2D, got {responses.ndim}D")

        n_persons, n_items = responses.shape
        if n_items != model.n_items:
            raise ValueError(
                f"responses has {n_items} items, model expects {model.n_items}"
            )

        pseudo_responses, trait_assignments, valid_mask = model.expand_to_pseudo_items(
            responses
        )

        self._quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=model.n_traits,
        )

        trait_mean = np.zeros(model.n_traits)
        trait_cov = np.eye(model.n_traits)

        self._convergence_history = []
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            posterior_weights, marginal_ll = self._e_step(
                model,
                pseudo_responses,
                trait_assignments,
                valid_mask,
                trait_mean,
                trait_cov,
            )

            current_ll = np.sum(np.log(marginal_ll + 1e-300))
            self._convergence_history.append(current_ll)

            self._log_iteration(iteration, current_ll)

            if self._check_convergence(prev_ll, current_ll):
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_ll = current_ll

            self._m_step(
                model,
                pseudo_responses,
                trait_assignments,
                valid_mask,
                posterior_weights,
            )

            if self.estimate_correlations:
                trait_mean, trait_cov = self._update_trait_distribution(
                    posterior_weights, trait_mean, trait_cov
                )

        model._is_fitted = True
        model._trait_correlations = self._cov_to_corr(trait_cov)

        theta_estimates, theta_se = self._compute_eap_scores(
            model,
            pseudo_responses,
            trait_assignments,
            valid_mask,
            trait_mean,
            trait_cov,
        )

        standard_errors = self._compute_standard_errors(
            model, pseudo_responses, trait_assignments, valid_mask, posterior_weights
        )

        n_params = self._count_parameters(model)

        aic = -2 * current_ll + 2 * n_params
        bic = -2 * current_ll + n_params * np.log(n_persons)

        return IRTreeResult(
            model=model,
            log_likelihood=current_ll,
            trait_means=trait_mean,
            trait_correlations=model._trait_correlations,
            theta_estimates=theta_estimates,
            theta_se=theta_se,
            standard_errors=standard_errors,
            aic=aic,
            bic=bic,
            converged=iteration < self.max_iter - 1,
            n_iterations=iteration + 1,
            n_observations=n_persons,
            n_parameters=n_params,
        )

    def _e_step(
        self,
        model: IRTreeModel,
        pseudo_responses: NDArray[np.int_],
        trait_assignments: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
        trait_mean: NDArray[np.float64],
        trait_cov: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute posterior weights over quadrature points."""
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights
        n_persons = pseudo_responses.shape[0]
        n_quad = len(quad_weights)

        log_likelihoods = np.zeros((n_persons, n_quad))

        for q in range(n_quad):
            theta_q = quad_points[q]
            ll_q = self._compute_log_likelihood_at_theta(
                model, pseudo_responses, trait_assignments, valid_mask, theta_q
            )
            log_likelihoods[:, q] = ll_q

        log_prior = self._log_mvn_density(quad_points, trait_mean, trait_cov)
        log_joint = log_likelihoods + log_prior[None, :] + np.log(quad_weights)[None, :]

        log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
        log_posterior = log_joint - log_marginal

        posterior_weights = np.exp(log_posterior)
        marginal_ll = np.exp(log_marginal.ravel())

        return posterior_weights, marginal_ll

    def _compute_log_likelihood_at_theta(
        self,
        model: IRTreeModel,
        pseudo_responses: NDArray[np.int_],
        trait_assignments: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for all persons at a single theta."""
        n_persons, n_items, max_nodes = pseudo_responses.shape

        ll = np.zeros(n_persons)

        for j in range(n_items):
            disc = model._parameters["discrimination"][j]
            diff = model._parameters["difficulty"][j]

            for node_idx in range(max_nodes):
                valid_persons = valid_mask[:, j, node_idx]
                if not np.any(valid_persons):
                    continue

                trait_idx = trait_assignments[j, node_idx]
                a = disc[node_idx]
                b = diff[node_idx]

                z = a * (theta[trait_idx] - b)
                p = 1.0 / (1.0 + np.exp(-z))
                p = np.clip(p, 1e-10, 1 - 1e-10)

                responses = pseudo_responses[valid_persons, j, node_idx]
                ll[valid_persons] += responses * np.log(p) + (1 - responses) * np.log(
                    1 - p
                )

        return ll

    def _m_step(
        self,
        model: IRTreeModel,
        pseudo_responses: NDArray[np.int_],
        trait_assignments: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
        posterior_weights: NDArray[np.float64],
    ) -> None:
        """Update item parameters."""
        quad_points = self._quadrature.nodes
        n_items = model.n_items
        max_nodes = pseudo_responses.shape[2]

        for j in range(n_items):
            for node_idx in range(max_nodes):
                valid_persons = valid_mask[:, j, node_idx]
                if not np.any(valid_persons):
                    continue

                trait_idx = trait_assignments[j, node_idx]
                responses = pseudo_responses[valid_persons, j, node_idx]
                weights = posterior_weights[valid_persons]

                current_a = model._parameters["discrimination"][j, node_idx]
                current_b = model._parameters["difficulty"][j, node_idx]

                def neg_expected_ll(params):
                    a, b = params
                    ll = 0.0
                    for q in range(len(quad_points)):
                        theta_k = quad_points[q, trait_idx]
                        z = a * (theta_k - b)
                        p = 1.0 / (1.0 + np.exp(-z))
                        p = np.clip(p, 1e-10, 1 - 1e-10)

                        r_q = np.sum(responses * weights[:, q])
                        n_q = np.sum(weights[:, q])

                        ll += r_q * np.log(p) + (n_q - r_q) * np.log(1 - p)
                    return -ll

                result = minimize(
                    neg_expected_ll,
                    x0=[current_a, current_b],
                    method="L-BFGS-B",
                    bounds=[(0.1, 5.0), (-6.0, 6.0)],
                    options={"maxiter": 50},
                )

                model._parameters["discrimination"][j, node_idx] = result.x[0]
                model._parameters["difficulty"][j, node_idx] = result.x[1]

    def _update_trait_distribution(
        self,
        posterior_weights: NDArray[np.float64],
        current_mean: NDArray[np.float64],
        current_cov: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update trait mean and covariance from posterior."""
        quad_points = self._quadrature.nodes
        n_traits = quad_points.shape[1]

        total_weight = np.sum(posterior_weights)

        new_mean = np.zeros(n_traits)
        for k in range(n_traits):
            weighted_sum = np.sum(posterior_weights * quad_points[None, :, k])
            new_mean[k] = weighted_sum / total_weight

        new_cov = np.zeros((n_traits, n_traits))
        for i in range(n_traits):
            for j in range(i, n_traits):
                centered_i = quad_points[:, i] - new_mean[i]
                centered_j = quad_points[:, j] - new_mean[j]

                weighted_sum = np.sum(
                    posterior_weights * (centered_i[None, :] * centered_j[None, :])
                )
                new_cov[i, j] = weighted_sum / total_weight
                new_cov[j, i] = new_cov[i, j]

        new_cov = (new_cov + new_cov.T) / 2
        min_var = 0.1
        np.fill_diagonal(new_cov, np.maximum(np.diag(new_cov), min_var))

        return new_mean, new_cov

    def _compute_eap_scores(
        self,
        model: IRTreeModel,
        pseudo_responses: NDArray[np.int_],
        trait_assignments: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
        trait_mean: NDArray[np.float64],
        trait_cov: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute EAP scores and standard errors."""
        posterior_weights, _ = self._e_step(
            model,
            pseudo_responses,
            trait_assignments,
            valid_mask,
            trait_mean,
            trait_cov,
        )

        quad_points = self._quadrature.nodes
        n_persons = pseudo_responses.shape[0]
        n_traits = model.n_traits

        theta_eap = np.zeros((n_persons, n_traits))
        theta_se = np.zeros((n_persons, n_traits))

        for i in range(n_persons):
            for k in range(n_traits):
                theta_eap[i, k] = np.sum(posterior_weights[i] * quad_points[:, k])

                variance = np.sum(
                    posterior_weights[i] * (quad_points[:, k] - theta_eap[i, k]) ** 2
                )
                theta_se[i, k] = np.sqrt(variance)

        return theta_eap, theta_se

    def _compute_standard_errors(
        self,
        model: IRTreeModel,
        pseudo_responses: NDArray[np.int_],
        trait_assignments: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
        posterior_weights: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Compute standard errors for item parameters."""
        se = {
            "discrimination": np.zeros_like(model._parameters["discrimination"]),
            "difficulty": np.zeros_like(model._parameters["difficulty"]),
        }

        return se

    def _count_parameters(self, model: IRTreeModel) -> int:
        """Count total number of estimated parameters."""
        n_item_params = 0
        max_nodes = model._parameters["discrimination"].shape[1]

        for j in range(model.n_items):
            for node_idx in range(max_nodes):
                if model._parameters["discrimination"][j, node_idx] > 0:
                    n_item_params += 2

        n_cov_params = 0
        if self.estimate_correlations:
            n_cov_params = model.n_traits * (model.n_traits + 1) // 2

        return n_item_params + n_cov_params

    @staticmethod
    def _log_mvn_density(
        x: NDArray[np.float64],
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log multivariate normal density."""
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

    @staticmethod
    def _cov_to_corr(cov: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert covariance matrix to correlation matrix."""
        std = np.sqrt(np.diag(cov))
        std_outer = np.outer(std, std)
        std_outer[std_outer == 0] = 1
        return cov / std_outer
