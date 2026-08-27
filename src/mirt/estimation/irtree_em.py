from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from mirt._core import sigmoid
from mirt._prior_mass import gaussian_log_quadrature_mass
from mirt.constants import PROB_EPSILON
from mirt.estimation.base import BaseEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.exceptions import MirtValidationError
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.irtree import IRTreeModel


@dataclass
class IRTreeResult:
    """Result from IRTree model estimation."""

    model: IRTreeModel
    log_likelihood: float
    trait_means: NDArray[np.float64]
    trait_covariance: NDArray[np.float64]
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
            var = self.trait_covariance[i, i]
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
        Whether to estimate the trait distribution when the model allows
        correlated traits
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
        if (
            isinstance(n_quadpts, bool)
            or not isinstance(n_quadpts, (int, np.integer))
            or n_quadpts < 1
        ):
            raise MirtValidationError(
                "n_quadpts must be a positive integer",
                parameter="n_quadpts",
                value=n_quadpts,
                expected=">= 1",
            )
        if not isinstance(estimate_correlations, (bool, np.bool_)):
            raise MirtValidationError(
                "estimate_correlations must be boolean",
                parameter="estimate_correlations",
                value=estimate_correlations,
            )

        self.n_quadpts = int(n_quadpts)
        self.estimate_correlations = bool(estimate_correlations)
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
        response_values = np.asarray(responses)
        pseudo_responses, trait_assignments, valid_mask = model.expand_to_pseudo_items(
            response_values
        )
        n_persons = response_values.shape[0]
        if n_persons == 0:
            raise ValueError("responses must contain at least one person")

        self._quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=model.n_traits,
        )

        trait_mean = np.zeros(model.n_traits)
        trait_cov = np.eye(model.n_traits)

        self._convergence_history = []
        prev_ll = -np.inf
        converged = False
        estimate_distribution = self.estimate_correlations and model.correlated_traits

        for iteration in range(self.max_iter):
            posterior_weights, log_marginal = self._e_step(
                model,
                pseudo_responses,
                trait_assignments,
                valid_mask,
                trait_mean,
                trait_cov,
                return_log=True,
            )

            current_ll = float(np.sum(log_marginal))
            self._convergence_history.append(current_ll)

            self._log_iteration(iteration, current_ll)

            if self._check_convergence(prev_ll, current_ll):
                converged = True
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

            if estimate_distribution:
                trait_mean, trait_cov = self._update_trait_distribution(
                    posterior_weights, trait_mean, trait_cov
                )

        posterior_weights, log_marginal = self._e_step(
            model,
            pseudo_responses,
            trait_assignments,
            valid_mask,
            trait_mean,
            trait_cov,
            return_log=True,
        )
        current_ll = float(np.sum(log_marginal))
        if not converged:
            self._convergence_history.append(current_ll)
            converged = self._check_convergence(prev_ll, current_ll)
        trait_correlations = self._cov_to_corr(trait_cov)
        model._is_fitted = True
        model._trait_correlations = (
            trait_correlations.copy() if model.correlated_traits else None
        )

        theta_estimates, theta_se = self._compute_eap_scores(
            model,
            pseudo_responses,
            trait_assignments,
            valid_mask,
            trait_mean,
            trait_cov,
            posterior_weights,
        )

        standard_errors = self._compute_standard_errors(
            model, pseudo_responses, trait_assignments, valid_mask, posterior_weights
        )

        n_params = self._count_parameters(model, estimate_distribution)

        aic = -2 * current_ll + 2 * n_params
        bic = -2 * current_ll + n_params * np.log(n_persons)

        return IRTreeResult(
            model=model,
            log_likelihood=current_ll,
            trait_means=trait_mean.copy(),
            trait_covariance=trait_cov.copy(),
            trait_correlations=trait_correlations.copy(),
            theta_estimates=theta_estimates,
            theta_se=theta_se,
            standard_errors=standard_errors,
            aic=aic,
            bic=bic,
            converged=converged,
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
        *,
        return_log: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute posterior weights and person marginal likelihoods."""
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights
        log_likelihoods = self._compute_log_likelihoods(
            model,
            pseudo_responses,
            trait_assignments,
            valid_mask,
            quad_points,
        )

        log_prior_mass = gaussian_log_quadrature_mass(
            quad_points, quad_weights, trait_mean, trait_cov
        )
        log_joint = log_likelihoods + log_prior_mass[None, :]

        log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
        log_posterior = log_joint - log_marginal

        posterior_weights = np.exp(log_posterior)
        marginal = log_marginal.ravel()
        if not return_log:
            marginal = np.exp(marginal)
        return posterior_weights, marginal

    @staticmethod
    def _compute_log_likelihoods(
        model: IRTreeModel,
        pseudo_responses: NDArray[np.int_],
        trait_assignments: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute all person-by-point log likelihoods with matrix products."""
        n_persons = pseudo_responses.shape[0]
        responses_flat = pseudo_responses.reshape(n_persons, -1)
        correct = (responses_flat == 1).astype(np.float64)
        valid = valid_mask.reshape(n_persons, -1).astype(np.float64)

        discrimination = model._parameters["discrimination"].reshape(-1)
        difficulty = model._parameters["difficulty"].reshape(-1)
        traits = trait_assignments.reshape(-1)
        logits = discrimination[None, :] * (theta[:, traits] - difficulty[None, :])
        probability = np.clip(
            sigmoid(logits),
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        return (
            correct @ np.log(probability).T
            + (valid - correct) @ np.log1p(-probability).T
        )

    def _compute_log_likelihood_at_theta(
        self,
        model: IRTreeModel,
        pseudo_responses: NDArray[np.int_],
        trait_assignments: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for all persons at a single theta."""
        theta_values = np.asarray(theta, dtype=np.float64).reshape(1, -1)
        return self._compute_log_likelihoods(
            model,
            pseudo_responses,
            trait_assignments,
            valid_mask,
            theta_values,
        )[:, 0]

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
        expected_correct, expected_total = self._expected_counts(
            pseudo_responses,
            valid_mask,
            posterior_weights,
        )

        for j in range(n_items):
            for node_idx in range(max_nodes):
                n_q = expected_total[j, node_idx]
                if not np.any(n_q > 0.0):
                    continue

                trait_idx = trait_assignments[j, node_idx]
                theta_values = quad_points[:, trait_idx]
                r_q = expected_correct[j, node_idx]

                current_a = model._parameters["discrimination"][j, node_idx]
                current_b = model._parameters["difficulty"][j, node_idx]

                def neg_expected_ll(
                    params: NDArray[np.float64],
                ) -> tuple[float, NDArray[np.float64]]:
                    a, b = params
                    centered = theta_values - b
                    probability = np.clip(
                        sigmoid(a * centered),
                        PROB_EPSILON,
                        1.0 - PROB_EPSILON,
                    )
                    expected_ll = np.sum(
                        r_q * np.log(probability) + (n_q - r_q) * np.log1p(-probability)
                    )
                    residual = r_q - n_q * probability
                    gradient = np.array(
                        [
                            -np.sum(residual * centered),
                            a * np.sum(residual),
                        ]
                    )
                    return -float(expected_ll), gradient

                result = minimize(
                    neg_expected_ll,
                    x0=[current_a, current_b],
                    method="L-BFGS-B",
                    jac=True,
                    bounds=[(0.1, 5.0), (-6.0, 6.0)],
                    options={"maxiter": 50},
                )

                if np.all(np.isfinite(result.x)):
                    model._parameters["discrimination"][j, node_idx] = result.x[0]
                    model._parameters["difficulty"][j, node_idx] = result.x[1]

    @staticmethod
    def _expected_counts(
        pseudo_responses: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
        posterior_weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Aggregate posterior-weighted correct and total node responses."""
        n_persons, n_items, max_nodes = pseudo_responses.shape
        correct = (pseudo_responses.reshape(n_persons, -1) == 1).astype(np.float64)
        valid = valid_mask.reshape(n_persons, -1).astype(np.float64)
        n_points = posterior_weights.shape[1]
        expected_correct = (correct.T @ posterior_weights).reshape(
            n_items, max_nodes, n_points
        )
        expected_total = (valid.T @ posterior_weights).reshape(
            n_items, max_nodes, n_points
        )
        return expected_correct, expected_total

    def _update_trait_distribution(
        self,
        posterior_weights: NDArray[np.float64],
        _current_mean: NDArray[np.float64],
        _current_cov: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update trait mean and covariance from posterior."""
        quad_points = self._quadrature.nodes
        point_weights = posterior_weights.sum(axis=0)
        total_weight = float(point_weights.sum())
        new_mean = point_weights @ quad_points / total_weight
        centered = quad_points - new_mean
        new_cov = (centered * point_weights[:, None]).T @ centered / total_weight
        new_cov = (new_cov + new_cov.T) * 0.5

        eigenvalues, eigenvectors = np.linalg.eigh(new_cov)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        new_cov = (eigenvectors * eigenvalues) @ eigenvectors.T
        min_var = 0.1
        variance_shortfall = np.maximum(min_var - np.diag(new_cov), 0.0)
        new_cov += np.diag(variance_shortfall)

        return new_mean, new_cov

    def _compute_eap_scores(
        self,
        model: IRTreeModel,
        pseudo_responses: NDArray[np.int_],
        trait_assignments: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
        trait_mean: NDArray[np.float64],
        trait_cov: NDArray[np.float64],
        posterior_weights: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute EAP scores and standard errors."""
        if posterior_weights is None:
            posterior_weights, _ = self._e_step(
                model,
                pseudo_responses,
                trait_assignments,
                valid_mask,
                trait_mean,
                trait_cov,
            )

        quad_points = self._quadrature.nodes
        theta_eap = posterior_weights @ quad_points
        second_moment = posterior_weights @ np.square(quad_points)
        theta_se = np.sqrt(np.maximum(second_moment - np.square(theta_eap), 0.0))
        return theta_eap, theta_se

    def _compute_standard_errors(
        self,
        model: IRTreeModel,
        pseudo_responses: NDArray[np.int_],
        trait_assignments: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
        posterior_weights: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Approximate item uncertainty from expected complete-data information."""
        se = {
            "discrimination": np.full_like(model._parameters["discrimination"], np.nan),
            "difficulty": np.full_like(model._parameters["difficulty"], np.nan),
        }

        _, expected_total = self._expected_counts(
            pseudo_responses,
            valid_mask,
            posterior_weights,
        )
        quad_points = self._quadrature.nodes
        n_items, max_nodes = model._parameters["discrimination"].shape

        for item_idx in range(n_items):
            for node_idx in range(max_nodes):
                n_q = expected_total[item_idx, node_idx]
                if not np.any(n_q > 0.0):
                    continue

                trait_idx = trait_assignments[item_idx, node_idx]
                theta_values = quad_points[:, trait_idx]
                discrimination = model._parameters["discrimination"][item_idx, node_idx]
                difficulty = model._parameters["difficulty"][item_idx, node_idx]
                centered = theta_values - difficulty
                probability = np.clip(
                    sigmoid(discrimination * centered),
                    PROB_EPSILON,
                    1.0 - PROB_EPSILON,
                )
                weight = n_q * probability * (1.0 - probability)
                score_a = centered
                score_b = -discrimination
                information = np.array(
                    [
                        [
                            np.sum(weight * np.square(score_a)),
                            np.sum(weight * score_a * score_b),
                        ],
                        [
                            np.sum(weight * score_a * score_b),
                            np.sum(weight * np.square(score_b)),
                        ],
                    ]
                )
                if np.linalg.matrix_rank(information) < 2:
                    continue
                covariance = np.linalg.pinv(information, rcond=1e-10)
                variance = np.diag(covariance)
                if np.all(np.isfinite(variance)) and np.all(variance > 0.0):
                    se["discrimination"][item_idx, node_idx] = np.sqrt(variance[0])
                    se["difficulty"][item_idx, node_idx] = np.sqrt(variance[1])

        return se

    def _count_parameters(self, model: IRTreeModel, estimate_distribution: bool) -> int:
        """Count total number of estimated parameters."""
        n_item_params = 2 * model.n_items * model.n_nodes
        if not estimate_distribution:
            return n_item_params
        n_mean_params = model.n_traits
        n_cov_params = model.n_traits * (model.n_traits + 1) // 2
        return n_item_params + n_mean_params + n_cov_params

    @staticmethod
    def _cov_to_corr(cov: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert covariance matrix to correlation matrix."""
        std = np.sqrt(np.diag(cov))
        std_outer = np.outer(std, std)
        std_outer[std_outer == 0] = 1
        return cov / std_outer
