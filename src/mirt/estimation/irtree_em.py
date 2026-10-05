from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from mirt._core import sigmoid
from mirt._prior_mass import gaussian_log_quadrature_mass
from mirt.constants import PROB_EPSILON
from mirt.estimation._dichotomous_objective import prepare_logistic_objective
from mirt.estimation._irtree_context import IRTreeFitContext
from mirt.estimation._posterior import normalize_log_posterior
from mirt.estimation.base import BaseEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.models.irtree import IRTreeModel

_MAX_IRTREE_SCRATCH_ENTRIES = 131_072


def _collapse_to_trait_grids(
    quad_points: NDArray[np.float64],
    traits: NDArray[np.int_],
    *counts: NDArray[np.float64],
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Sum per-node quadrature counts over points sharing the node's trait value.

    A binary node depends on the quadrature grid only through its own trait's
    coordinate, so merging grid points with equal coordinates leaves every
    node objective, gradient and information matrix unchanged. On a product
    grid this shrinks ``n_quadpts ** n_traits`` points to ``n_quadpts``.

    Parameters
    ----------
    quad_points : NDArray
        Quadrature nodes with shape ``(n_points, n_traits)``.
    traits : NDArray
        Trait index of each node with shape ``(n_nodes,)``.
    *counts : NDArray
        Count arrays with shape ``(n_nodes, n_points)``.

    Returns
    -------
    tuple
        ``(points, collapsed)``: per-node trait values with shape
        ``(n_nodes, n_values)`` and the matching summed counts. Traits with
        fewer distinct values are padded with zero counts.
    """
    orders = np.argsort(quad_points, axis=0, kind="stable")
    # On sorted coordinates the first index of each distinct value starts its run.
    grids = [
        np.unique(quad_points[order, trait], return_index=True)
        for trait, order in enumerate(orders.T)
    ]
    width = max(grid.size for grid, _ in grids)
    points = np.zeros((traits.size, width))
    collapsed = [np.zeros((traits.size, width)) for _ in counts]
    for trait, (grid, starts) in enumerate(grids):
        rows = np.flatnonzero(traits == trait)
        if not rows.size:
            continue
        points[rows, : grid.size] = grid
        for source, target in zip(counts, collapsed, strict=True):
            target[rows, : grid.size] = np.add.reduceat(
                source[rows][:, orders[:, trait]], starts, axis=1
            )
    return points, collapsed


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
        self._fit_context: IRTreeFitContext | None = None

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

        previous_context = self._fit_context
        with IRTreeFitContext(pseudo_responses, valid_mask) as context:
            self._fit_context = context
            try:
                return self._fit_prepared(
                    model, pseudo_responses, trait_assignments, valid_mask
                )
            finally:
                self._fit_context = previous_context

    def _fit_prepared(
        self,
        model: IRTreeModel,
        pseudo_responses: NDArray[np.int_],
        trait_assignments: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
    ) -> IRTreeResult:
        n_persons = pseudo_responses.shape[0]

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
        posterior_weights: NDArray[np.float64] | None = None

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
            posterior_weights = None

        if not (converged and self._uses_default_method("_e_step")):
            if converged:
                posterior_weights = None
            posterior_weights, log_marginal = self._e_step(
                model,
                pseudo_responses,
                trait_assignments,
                valid_mask,
                trait_mean,
                trait_cov,
                return_log=True,
            )
        assert posterior_weights is not None
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
        if self._uses_default_method("_compute_log_likelihoods"):
            log_likelihoods = self._compute_log_likelihoods(
                model,
                pseudo_responses,
                trait_assignments,
                valid_mask,
                quad_points,
                context=self._response_context(pseudo_responses, valid_mask),
            )
        else:
            log_likelihoods = np.array(
                self._compute_log_likelihoods(
                    model, pseudo_responses, trait_assignments, valid_mask, quad_points
                ),
                dtype=np.float64,
                copy=True,
            )

        log_prior_mass = gaussian_log_quadrature_mass(
            quad_points, quad_weights, trait_mean, trait_cov
        )
        posterior_weights, marginal = normalize_log_posterior(
            log_likelihoods, log_prior_mass
        )
        if not return_log:
            marginal = np.exp(marginal)
        return posterior_weights, marginal

    def _uses_default_method(self, name: str) -> bool:
        return (
            type(self) is IRTreeEMEstimator
            and name not in vars(self)
            and getattr(IRTreeEMEstimator, name) is _DEFAULT_IRTREE_METHODS[name]
        )

    def _response_context(
        self,
        pseudo_responses: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
    ) -> IRTreeFitContext:
        context = self._fit_context
        if (
            context is not None
            and context.pseudo_responses is pseudo_responses
            and context.valid_mask is valid_mask
        ):
            return context
        return IRTreeFitContext(pseudo_responses, valid_mask)

    @staticmethod
    def _compute_log_likelihoods(
        model: IRTreeModel,
        pseudo_responses: NDArray[np.int_],
        trait_assignments: NDArray[np.int_],
        valid_mask: NDArray[np.bool_],
        theta: NDArray[np.float64],
        *,
        context: IRTreeFitContext | None = None,
    ) -> NDArray[np.float64]:
        """Compute all person-by-point log likelihoods with matrix products."""
        n_persons = pseudo_responses.shape[0]
        context = context or IRTreeFitContext(pseudo_responses, valid_mask)
        discrimination = model._parameters["discrimination"].reshape(-1)
        difficulty = model._parameters["difficulty"].reshape(-1)
        traits = trait_assignments.reshape(-1)
        n_points = theta.shape[0]
        likelihood = np.zeros((n_persons, n_points))
        node_chunk = max(1, _MAX_IRTREE_SCRATCH_ENTRIES // max(1, n_points))
        for first in range(0, discrimination.size, node_chunk):
            last = min(first + node_chunk, discrimination.size)
            logits = discrimination[None, first:last] * (
                theta[:, traits[first:last]] - difficulty[None, first:last]
            )
            probability = np.clip(sigmoid(logits), PROB_EPSILON, 1.0 - PROB_EPSILON)
            log_failure = np.log1p(-probability)
            log_odds = np.log(probability) - log_failure
            row_chunk = max(
                1,
                _MAX_IRTREE_SCRATCH_ENTRIES // max(n_points, 2 * (last - first)),
            )
            for start in range(0, n_persons, row_chunk):
                stop = min(start + row_chunk, n_persons)
                correct, observed = context.node_components(start, stop, first, last)
                block = likelihood[start:stop]
                block += correct @ log_odds.T
                block += observed @ log_failure.T
        return likelihood

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
        if self._uses_default_method("_expected_counts"):
            expected_correct, expected_total = self._expected_counts(
                pseudo_responses,
                valid_mask,
                posterior_weights,
                context=self._response_context(pseudo_responses, valid_mask),
            )
        else:
            expected_correct, expected_total = self._expected_counts(
                pseudo_responses,
                valid_mask,
                posterior_weights,
            )

        node_points, (node_correct, node_total) = _collapse_to_trait_grids(
            quad_points,
            trait_assignments.reshape(-1),
            expected_correct.reshape(-1, quad_points.shape[0]),
            expected_total.reshape(-1, quad_points.shape[0]),
        )
        bounds = [(0.1, 5.0), (-6.0, 6.0)]
        for j in range(n_items):
            for node_idx in range(max_nodes):
                flat_idx = j * max_nodes + node_idx
                n_q = node_total[flat_idx]
                if not np.any(n_q > 0.0):
                    continue

                current_a = model._parameters["discrimination"][j, node_idx]
                current_b = model._parameters["difficulty"][j, node_idx]

                neg_expected_ll = prepare_logistic_objective(
                    node_points[flat_idx, :, None],
                    n_q,
                    node_correct[flat_idx],
                    PROB_EPSILON,
                    bounds=bounds,
                )
                if neg_expected_ll is None:
                    raise RuntimeError("Unable to prepare IRTree node objective")

                result = minimize(
                    neg_expected_ll,
                    x0=[current_a, current_b],
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
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
        *,
        context: IRTreeFitContext | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Aggregate posterior-weighted correct and total node responses."""
        _, n_items, max_nodes = pseudo_responses.shape
        context = context or IRTreeFitContext(pseudo_responses, valid_mask)
        n_points = posterior_weights.shape[1]
        correct, total = context.expected_counts(posterior_weights)
        expected_correct = correct.reshape(n_items, max_nodes, n_points)
        expected_total = total.reshape(n_items, max_nodes, n_points)
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

        if self._uses_default_method("_expected_counts"):
            expected_total = self._response_context(
                pseudo_responses, valid_mask
            ).expected_totals(posterior_weights)
        else:
            _, counts = self._expected_counts(
                pseudo_responses, valid_mask, posterior_weights
            )
            expected_total = counts.reshape(-1, counts.shape[-1])
        node_points, (node_total,) = _collapse_to_trait_grids(
            self._quadrature.nodes,
            trait_assignments.reshape(-1),
            expected_total,
        )
        slopes = model._parameters["discrimination"].reshape(-1)
        difficulties = model._parameters["difficulty"].reshape(-1)
        slope_se = se["discrimination"].reshape(-1)
        difficulty_se = se["difficulty"].reshape(-1)
        chunk_size = max(1, _MAX_IRTREE_SCRATCH_ENTRIES // node_points.shape[1])
        for start in range(0, slopes.size, chunk_size):
            stop = min(start + chunk_size, slopes.size)
            centered = node_points[start:stop].T - difficulties[None, start:stop]
            probability = np.clip(
                sigmoid(slopes[None, start:stop] * centered),
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            )
            weight = node_total[start:stop].T * probability * (1.0 - probability)
            score_b = -slopes[start:stop]
            information = np.empty((stop - start, 2, 2))
            information[:, 0, 0] = np.sum(weight * np.square(centered), axis=0)
            information[:, 0, 1] = np.sum(weight * centered * score_b, axis=0)
            information[:, 1, 0] = information[:, 0, 1]
            information[:, 1, 1] = np.sum(weight * np.square(score_b), axis=0)
            selected = np.flatnonzero(np.linalg.matrix_rank(information) >= 2)
            if not selected.size:
                continue
            covariance = np.linalg.pinv(information[selected], rcond=1e-10)
            variance = np.diagonal(covariance, axis1=1, axis2=2)
            finite = np.all(np.isfinite(variance) & (variance > 0.0), axis=1)
            indices = start + selected[finite]
            slope_se[indices] = np.sqrt(variance[finite, 0])
            difficulty_se[indices] = np.sqrt(variance[finite, 1])

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


_DEFAULT_IRTREE_METHODS = {
    name: getattr(IRTreeEMEstimator, name)
    for name in ("_e_step", "_compute_log_likelihoods", "_expected_counts")
}
