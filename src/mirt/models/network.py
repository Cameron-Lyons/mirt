"""Network Psychometrics Models.

This module provides:
- IsingModel for binary network models
- GaussianGraphicalModel for continuous/ordinal data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON


@dataclass
class NetworkResult:
    """Results from network model estimation.

    Attributes
    ----------
    model : IsingModel or GaussianGraphicalModel
        Fitted model.
    log_pseudo_likelihood : float
        Log-pseudolikelihood at convergence.
    n_iterations : int
        Number of iterations.
    converged : bool
        Whether estimation converged.
    """

    model: IsingModel | GaussianGraphicalModel
    log_pseudo_likelihood: float
    n_iterations: int
    converged: bool


class IsingModel:
    """Binary network model for item responses.

    The Ising model represents binary variables as nodes in a network,
    with thresholds (main effects) and pairwise interactions (edges).

    P(X = x) ∝ exp(Σ_i τ_i x_i + Σ_{i<j} ω_{ij} x_i x_j)

    Parameters
    ----------
    n_nodes : int
        Number of nodes (items/variables).
    node_names : list of str, optional
        Names for nodes.

    Attributes
    ----------
    thresholds : NDArray
        Node threshold parameters (n_nodes,).
    interactions : NDArray
        Pairwise interaction matrix (n_nodes, n_nodes), symmetric.
    """

    def __init__(
        self,
        n_nodes: int,
        node_names: list[str] | None = None,
    ) -> None:
        if n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")

        self._n_nodes = n_nodes
        self._node_names = node_names or [f"X{i}" for i in range(n_nodes)]

        if len(self._node_names) != n_nodes:
            raise ValueError(
                f"node_names length ({len(self._node_names)}) "
                f"must match n_nodes ({n_nodes})"
            )

        self._thresholds = np.zeros(n_nodes, dtype=np.float64)
        self._interactions = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        self._is_fitted = False

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def node_names(self) -> list[str]:
        return list(self._node_names)

    @property
    def thresholds(self) -> NDArray[np.float64]:
        return self._thresholds.copy()

    @property
    def interactions(self) -> NDArray[np.float64]:
        return self._interactions.copy()

    def set_thresholds(self, thresholds: NDArray[np.float64]) -> Self:
        thresholds = np.asarray(thresholds, dtype=np.float64)
        if thresholds.shape != (self._n_nodes,):
            raise ValueError(
                f"thresholds shape {thresholds.shape} != ({self._n_nodes},)"
            )
        self._thresholds = thresholds
        return self

    def set_interactions(self, interactions: NDArray[np.float64]) -> Self:
        interactions = np.asarray(interactions, dtype=np.float64)
        if interactions.shape != (self._n_nodes, self._n_nodes):
            raise ValueError(
                f"interactions shape {interactions.shape} != "
                f"({self._n_nodes}, {self._n_nodes})"
            )

        interactions = (interactions + interactions.T) / 2
        np.fill_diagonal(interactions, 0)

        self._interactions = interactions
        return self

    def conditional_probability(
        self,
        node_idx: int,
        other_values: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Compute P(X_i = 1 | X_{-i}).

        Parameters
        ----------
        node_idx : int
            Index of node to compute probability for.
        other_values : NDArray
            Values of other nodes (n_samples, n_nodes).

        Returns
        -------
        NDArray
            Probability X_i = 1 given other nodes (n_samples,).
        """
        other_values = np.asarray(other_values)
        if other_values.ndim == 1:
            other_values = other_values.reshape(1, -1)

        linear_pred = self._thresholds[node_idx]
        linear_pred = linear_pred + np.dot(
            other_values, self._interactions[node_idx, :]
        )

        return 1 / (1 + np.exp(-linear_pred))

    def pseudo_likelihood(self, responses: NDArray[np.int_]) -> float:
        """Compute log-pseudolikelihood.

        The pseudolikelihood is the product of full conditionals.

        Parameters
        ----------
        responses : NDArray
            Binary response matrix (n_samples, n_nodes).

        Returns
        -------
        float
            Log-pseudolikelihood.
        """
        responses = np.asarray(responses)
        n_samples = responses.shape[0]

        log_psl = 0.0
        for i in range(self._n_nodes):
            p_i = self.conditional_probability(i, responses)
            p_i = np.clip(p_i, PROB_EPSILON, 1 - PROB_EPSILON)
            log_psl += np.sum(
                responses[:, i] * np.log(p_i) + (1 - responses[:, i]) * np.log(1 - p_i)
            )

        return log_psl / n_samples

    def sample(
        self,
        n_samples: int,
        n_burnin: int = 1000,
        seed: int | None = None,
    ) -> NDArray[np.int_]:
        """Generate samples using Gibbs sampling.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        n_burnin : int
            Number of burn-in iterations.
        seed : int, optional
            Random seed.

        Returns
        -------
        NDArray
            Binary samples (n_samples, n_nodes).
        """
        rng = np.random.default_rng(seed)

        current = rng.binomial(1, 0.5, self._n_nodes)
        samples = np.zeros((n_samples, self._n_nodes), dtype=np.int_)

        for t in range(n_burnin + n_samples):
            for i in range(self._n_nodes):
                p_i = self.conditional_probability(i, current.reshape(1, -1))[0]
                current[i] = rng.binomial(1, p_i)

            if t >= n_burnin:
                samples[t - n_burnin] = current

        return samples

    def edge_weights(self) -> NDArray[np.float64]:
        """Get edge weight matrix (upper triangle only)."""
        return np.triu(self._interactions, k=1)

    def degree_centrality(self) -> NDArray[np.float64]:
        """Compute degree centrality for each node."""
        return np.sum(np.abs(self._interactions), axis=1) / (self._n_nodes - 1)

    def expected_influence(self) -> NDArray[np.float64]:
        """Compute expected influence for each node.

        Sum of all edge weights connected to each node.
        """
        return np.sum(self._interactions, axis=1)

    def copy(self) -> Self:
        new_model = IsingModel(
            n_nodes=self._n_nodes,
            node_names=list(self._node_names),
        )
        new_model._thresholds = self._thresholds.copy()
        new_model._interactions = self._interactions.copy()
        new_model._is_fitted = self._is_fitted
        return new_model


class GaussianGraphicalModel:
    """Gaussian Graphical Model for continuous/ordinal data.

    Models the conditional independence structure using a precision
    matrix (inverse covariance). Partial correlations indicate
    direct associations after controlling for all other variables.

    Parameters
    ----------
    n_nodes : int
        Number of nodes (variables).
    node_names : list of str, optional
        Names for nodes.

    Attributes
    ----------
    means : NDArray
        Node means (n_nodes,).
    precision_matrix : NDArray
        Precision matrix (n_nodes, n_nodes).
    """

    def __init__(
        self,
        n_nodes: int,
        node_names: list[str] | None = None,
    ) -> None:
        if n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")

        self._n_nodes = n_nodes
        self._node_names = node_names or [f"X{i}" for i in range(n_nodes)]

        if len(self._node_names) != n_nodes:
            raise ValueError(
                f"node_names length ({len(self._node_names)}) "
                f"must match n_nodes ({n_nodes})"
            )

        self._means = np.zeros(n_nodes, dtype=np.float64)
        self._precision = np.eye(n_nodes, dtype=np.float64)
        self._is_fitted = False

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def node_names(self) -> list[str]:
        return list(self._node_names)

    @property
    def means(self) -> NDArray[np.float64]:
        return self._means.copy()

    @property
    def precision_matrix(self) -> NDArray[np.float64]:
        return self._precision.copy()

    @property
    def covariance_matrix(self) -> NDArray[np.float64]:
        """Covariance matrix (inverse of precision)."""
        return np.linalg.inv(self._precision)

    def set_means(self, means: NDArray[np.float64]) -> Self:
        means = np.asarray(means, dtype=np.float64)
        if means.shape != (self._n_nodes,):
            raise ValueError(f"means shape {means.shape} != ({self._n_nodes},)")
        self._means = means
        return self

    def set_precision_matrix(self, precision: NDArray[np.float64]) -> Self:
        precision = np.asarray(precision, dtype=np.float64)
        if precision.shape != (self._n_nodes, self._n_nodes):
            raise ValueError(
                f"precision shape {precision.shape} != "
                f"({self._n_nodes}, {self._n_nodes})"
            )

        precision = (precision + precision.T) / 2

        eigvals = np.linalg.eigvalsh(precision)
        if np.any(eigvals <= 0):
            raise ValueError("precision matrix must be positive definite")

        self._precision = precision
        return self

    def partial_correlations(self) -> NDArray[np.float64]:
        """Compute partial correlation matrix.

        Partial correlations are standardized off-diagonal elements
        of the negative precision matrix.

        Returns
        -------
        NDArray
            Partial correlation matrix (n_nodes, n_nodes).
        """
        d = np.sqrt(np.diag(self._precision))
        d_inv = 1 / d
        partial = -self._precision * np.outer(d_inv, d_inv)
        np.fill_diagonal(partial, 1.0)
        return partial

    def conditional_mean(
        self,
        node_idx: int,
        other_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute E[X_i | X_{-i}].

        Parameters
        ----------
        node_idx : int
            Index of node.
        other_values : NDArray
            Values of other nodes (n_samples, n_nodes).

        Returns
        -------
        NDArray
            Conditional mean (n_samples,).
        """
        other_values = np.asarray(other_values)
        if other_values.ndim == 1:
            other_values = other_values.reshape(1, -1)

        prec_ii = self._precision[node_idx, node_idx]
        prec_i_neg_i = np.delete(self._precision[node_idx, :], node_idx)
        other_centered = np.delete(other_values - self._means, node_idx, axis=1)

        cond_mean = (
            self._means[node_idx] - np.dot(other_centered, prec_i_neg_i) / prec_ii
        )
        return cond_mean

    def conditional_variance(self, node_idx: int) -> float:
        """Compute Var[X_i | X_{-i}]."""
        return 1 / self._precision[node_idx, node_idx]

    def log_likelihood(self, data: NDArray[np.float64]) -> float:
        """Compute log-likelihood.

        Parameters
        ----------
        data : NDArray
            Data matrix (n_samples, n_nodes).

        Returns
        -------
        float
            Log-likelihood.
        """
        data = np.asarray(data)
        n_samples = data.shape[0]

        centered = data - self._means
        sign, log_det = np.linalg.slogdet(self._precision)

        if sign <= 0:
            return -np.inf

        quad_form = np.sum(centered @ self._precision * centered)

        ll = (
            -0.5 * n_samples * self._n_nodes * np.log(2 * np.pi)
            + 0.5 * n_samples * log_det
            - 0.5 * quad_form
        )

        return ll

    def sample(
        self,
        n_samples: int,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Generate samples from the model.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        seed : int, optional
            Random seed.

        Returns
        -------
        NDArray
            Samples (n_samples, n_nodes).
        """
        rng = np.random.default_rng(seed)
        cov = self.covariance_matrix
        return rng.multivariate_normal(self._means, cov, size=n_samples)

    def edge_weights(self) -> NDArray[np.float64]:
        """Get edge weight matrix (partial correlations, upper triangle)."""
        partial = self.partial_correlations()
        return np.triu(partial, k=1)

    def degree_centrality(self) -> NDArray[np.float64]:
        """Compute degree centrality based on partial correlations."""
        partial = self.partial_correlations()
        np.fill_diagonal(partial, 0)
        return np.sum(np.abs(partial), axis=1) / (self._n_nodes - 1)

    def copy(self) -> Self:
        new_model = GaussianGraphicalModel(
            n_nodes=self._n_nodes,
            node_names=list(self._node_names),
        )
        new_model._means = self._means.copy()
        new_model._precision = self._precision.copy()
        new_model._is_fitted = self._is_fitted
        return new_model


def fit_ising(
    responses: NDArray[np.int_],
    regularization: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
) -> tuple[IsingModel, float]:
    """Fit Ising model using pseudolikelihood with L1 regularization.

    Parameters
    ----------
    responses : NDArray
        Binary response matrix (n_samples, n_nodes).
    regularization : float
        L1 regularization parameter (LASSO-like sparsity).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    tuple
        (fitted_model, log_pseudo_likelihood)
    """
    responses = np.asarray(responses)
    n_samples, n_nodes = responses.shape

    model = IsingModel(n_nodes=n_nodes)

    thresholds = np.zeros(n_nodes)
    interactions = np.zeros((n_nodes, n_nodes))

    for iteration in range(max_iter):
        prev_thresholds = thresholds.copy()
        prev_interactions = interactions.copy()

        for i in range(n_nodes):
            y = responses[:, i]
            X = np.column_stack([np.ones(n_samples), responses])
            X[:, i + 1] = 0

            p = 1 / (1 + np.exp(-(thresholds[i] + responses @ interactions[i, :])))
            p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)

            grad_threshold = np.mean(y - p)
            grad_interactions = np.mean((y - p)[:, None] * responses, axis=0)
            grad_interactions[i] = 0

            step_size = 0.5
            thresholds[i] += step_size * grad_threshold

            for j in range(n_nodes):
                if j != i:
                    new_val = interactions[i, j] + step_size * grad_interactions[j]
                    if regularization > 0:
                        new_val = np.sign(new_val) * max(
                            0, abs(new_val) - step_size * regularization
                        )
                    interactions[i, j] = new_val
                    interactions[j, i] = new_val

        model.set_thresholds(thresholds)
        model.set_interactions(interactions)
        psl = model.pseudo_likelihood(responses)

        if verbose:
            print(f"Iteration {iteration + 1}: PSL = {psl:.4f}")

        thresh_change = np.max(np.abs(thresholds - prev_thresholds))
        inter_change = np.max(np.abs(interactions - prev_interactions))

        if max(thresh_change, inter_change) < tol:
            break

    model._is_fitted = True
    return model, model.pseudo_likelihood(responses)


def fit_ggm(
    data: NDArray[np.float64],
    regularization: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> tuple[GaussianGraphicalModel, float]:
    """Fit Gaussian Graphical Model with graphical LASSO.

    Parameters
    ----------
    data : NDArray
        Data matrix (n_samples, n_nodes).
    regularization : float
        L1 regularization parameter.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    tuple
        (fitted_model, log_likelihood)
    """
    data = np.asarray(data, dtype=np.float64)
    n_samples, n_nodes = data.shape

    means = np.mean(data, axis=0)
    sample_cov = np.cov(data, rowvar=False, ddof=0)

    if sample_cov.ndim == 0:
        sample_cov = np.array([[sample_cov]])

    precision = np.linalg.inv(sample_cov + regularization * np.eye(n_nodes))

    if regularization > 0:
        for iteration in range(max_iter):
            prev_precision = precision.copy()

            cov = np.linalg.inv(precision)

            for j in range(n_nodes):
                idx_not_j = np.arange(n_nodes) != j
                W_jj = cov[j, j]
                s_j = sample_cov[idx_not_j, j]
                W_not_j = cov[np.ix_(idx_not_j, idx_not_j)]

                try:
                    beta = np.linalg.solve(W_not_j, s_j)
                except np.linalg.LinAlgError:
                    beta = np.zeros(n_nodes - 1)

                for k in range(len(beta)):
                    beta[k] = np.sign(beta[k]) * max(
                        0, abs(beta[k]) - regularization / (2 * n_samples)
                    )

                cov[j, idx_not_j] = W_jj * beta
                cov[idx_not_j, j] = W_jj * beta

            try:
                precision = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                precision = np.linalg.pinv(cov)

            if verbose:
                print(f"Iteration {iteration + 1}")

            if np.max(np.abs(precision - prev_precision)) < tol:
                break

    model = GaussianGraphicalModel(n_nodes=n_nodes)
    model.set_means(means)

    try:
        model.set_precision_matrix(precision)
    except ValueError:
        precision = precision + 0.01 * np.eye(n_nodes)
        model.set_precision_matrix(precision)

    model._is_fitted = True
    return model, model.log_likelihood(data)


def compare_networks(
    model1: IsingModel | GaussianGraphicalModel,
    model2: IsingModel | GaussianGraphicalModel,
) -> dict:
    """Compare two network models.

    Parameters
    ----------
    model1, model2 : IsingModel or GaussianGraphicalModel
        Models to compare.

    Returns
    -------
    dict
        Comparison metrics including edge correlation, degree correlation.
    """
    if not isinstance(model1, type(model2)):
        raise ValueError("Models must be of the same type")

    if model1.n_nodes != model2.n_nodes:
        raise ValueError("Models must have the same number of nodes")

    def _safe_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        """Return Pearson correlation, or NaN when undefined."""
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        if x.size != y.size or x.size < 2:
            return np.nan

        eps = np.finfo(np.float64).eps
        if np.std(x) <= eps or np.std(y) <= eps:
            return np.nan

        return float(np.corrcoef(x, y)[0, 1])

    edges1 = model1.edge_weights().flatten()
    edges2 = model2.edge_weights().flatten()

    edge_corr = _safe_correlation(edges1, edges2)

    deg1 = model1.degree_centrality()
    deg2 = model2.degree_centrality()
    deg_corr = _safe_correlation(deg1, deg2)

    edge_diff = np.abs(edges1 - edges2)

    return {
        "edge_correlation": edge_corr,
        "degree_correlation": deg_corr,
        "mean_edge_difference": np.mean(edge_diff),
        "max_edge_difference": np.max(edge_diff),
    }
