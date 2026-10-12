"""Network psychometrics models for binary and continuous observations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from mirt.exceptions import MirtDataError, MirtValidationError

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _validate_node_metadata(
    n_nodes: int, node_names: list[str] | None
) -> tuple[int, list[str]]:
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)):
        raise MirtValidationError(
            "n_nodes must be an integer", parameter="n_nodes", value=n_nodes
        )
    n_nodes = int(n_nodes)
    if n_nodes < 2:
        raise MirtValidationError(
            "n_nodes must be at least 2",
            parameter="n_nodes",
            value=n_nodes,
            expected=">= 2",
        )

    names = (
        [f"X{i}" for i in range(n_nodes)] if node_names is None else list(node_names)
    )
    if len(names) != n_nodes:
        raise MirtValidationError(
            f"node_names length ({len(names)}) must match n_nodes ({n_nodes})",
            parameter="node_names",
        )
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise MirtValidationError(
            "node_names must contain non-empty strings", parameter="node_names"
        )
    names = [name.strip() for name in names]
    if len(names) != len(set(names)):
        raise MirtValidationError(
            "node_names must be unique", parameter="node_names", value=names
        )
    return n_nodes, names


def _validate_node_index(node_idx: int, n_nodes: int) -> int:
    if isinstance(node_idx, bool) or not isinstance(node_idx, (int, np.integer)):
        raise MirtValidationError(
            "node_idx must be an integer", parameter="node_idx", value=node_idx
        )
    node_idx = int(node_idx)
    if node_idx < 0 or node_idx >= n_nodes:
        raise IndexError(f"Node index {node_idx} out of range [0, {n_nodes})")
    return node_idx


def _validate_count(name: str, value: int, *, allow_zero: bool) -> int:
    minimum = 0 if allow_zero else 1
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise MirtValidationError(
            f"{name} must be a {qualifier} integer", parameter=name, value=value
        )
    return int(value)


def _validate_optimizer_options(
    regularization: float, max_iter: int, tol: float
) -> tuple[float, int, float]:
    regularization = float(regularization)
    tol = float(tol)
    max_iter = _validate_count("max_iter", max_iter, allow_zero=False)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise MirtValidationError(
            "regularization must be finite and non-negative",
            parameter="regularization",
            value=regularization,
        )
    if not np.isfinite(tol) or tol <= 0.0:
        raise MirtValidationError(
            "tol must be finite and positive", parameter="tol", value=tol
        )
    return regularization, max_iter, tol


def _validate_binary_responses(
    responses: NDArray[np.int_],
    *,
    n_nodes: int | None = None,
    minimum_observations: int = 1,
) -> IntArray:
    values = np.asarray(responses)
    if values.ndim != 2:
        raise MirtDataError(
            "responses must be a two-dimensional matrix",
            value=values.ndim,
            expected="2 dimensions",
        )
    if n_nodes is not None and values.shape[1] != n_nodes:
        raise MirtDataError(
            f"responses has {values.shape[1]} nodes, expected {n_nodes}",
            n_items=values.shape[1],
        )
    if values.shape[1] < 2:
        raise MirtDataError("responses must contain at least two nodes")
    if values.shape[0] < minimum_observations:
        raise MirtDataError(
            f"responses must contain at least {minimum_observations} observations"
        )
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise MirtDataError("responses must contain finite binary values")
    integer_values = values.astype(np.int64)
    if not np.array_equal(values, integer_values) or np.any(
        (integer_values != 0) & (integer_values != 1)
    ):
        raise MirtDataError("responses must contain only 0 and 1")
    return integer_values


def _validate_continuous_data(
    data: NDArray[np.float64],
    *,
    n_nodes: int | None = None,
    minimum_observations: int = 1,
) -> FloatArray:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise MirtDataError(
            "data must be a two-dimensional matrix",
            value=values.ndim,
            expected="2 dimensions",
        )
    if n_nodes is not None and values.shape[1] != n_nodes:
        raise MirtDataError(
            f"data has {values.shape[1]} nodes, expected {n_nodes}",
            n_items=values.shape[1],
        )
    if values.shape[1] < 2:
        raise MirtDataError("data must contain at least two nodes")
    if values.shape[0] < minimum_observations:
        raise MirtDataError(
            f"data must contain at least {minimum_observations} observations"
        )
    if not np.all(np.isfinite(values)):
        raise MirtDataError("data must contain only finite values")
    return values


def _stable_sigmoid(values: FloatArray) -> FloatArray:
    return np.exp(-np.logaddexp(0.0, -values))


def _soft_threshold(values: FloatArray, threshold: float) -> FloatArray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


@dataclass
class NetworkResult:
    """Summary of a network model fit."""

    model: IsingModel | GaussianGraphicalModel
    log_pseudo_likelihood: float
    n_iterations: int
    converged: bool


class IsingModel:
    """Binary network with thresholds and symmetric pairwise interactions.

    The model uses the 0/1 parameterization

    ``P(X=x) ∝ exp(thresholds @ x + 0.5 * x.T @ interactions @ x)``.
    """

    def __init__(
        self,
        n_nodes: int,
        node_names: list[str] | None = None,
    ) -> None:
        self._n_nodes, self._node_names = _validate_node_metadata(n_nodes, node_names)
        self._thresholds = np.zeros(self._n_nodes, dtype=np.float64)
        self._interactions = np.zeros((self._n_nodes, self._n_nodes), dtype=np.float64)
        self._is_fitted = False
        self._n_iterations = 0
        self._converged = False
        self._objective_history: list[float] = []
        self._log_partition_cache: float | None = None
        self._exact_cdf_cache: FloatArray | None = None

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def node_names(self) -> list[str]:
        return self._node_names.copy()

    @property
    def thresholds(self) -> FloatArray:
        return self._thresholds.copy()

    @property
    def interactions(self) -> FloatArray:
        return self._interactions.copy()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_iterations(self) -> int:
        return self._n_iterations

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def objective_history(self) -> FloatArray:
        return np.asarray(self._objective_history, dtype=np.float64)

    def set_thresholds(self, thresholds: NDArray[np.float64]) -> Self:
        values = np.asarray(thresholds, dtype=np.float64)
        if values.shape != (self._n_nodes,):
            raise MirtValidationError(
                f"thresholds shape {values.shape} != ({self._n_nodes},)",
                parameter="thresholds",
            )
        if not np.all(np.isfinite(values)):
            raise MirtValidationError(
                "thresholds must contain only finite values",
                parameter="thresholds",
            )
        self._thresholds = values.copy()
        self._log_partition_cache = None
        self._exact_cdf_cache = None
        return self

    def set_interactions(self, interactions: NDArray[np.float64]) -> Self:
        values = np.asarray(interactions, dtype=np.float64)
        expected = (self._n_nodes, self._n_nodes)
        if values.shape != expected:
            raise MirtValidationError(
                f"interactions shape {values.shape} != {expected}",
                parameter="interactions",
            )
        if not np.all(np.isfinite(values)):
            raise MirtValidationError(
                "interactions must contain only finite values",
                parameter="interactions",
            )
        values = (values + values.T) / 2.0
        np.fill_diagonal(values, 0.0)
        self._interactions = values
        self._log_partition_cache = None
        self._exact_cdf_cache = None
        return self

    def _linear_predictors(self, responses: IntArray) -> FloatArray:
        return self._thresholds + responses @ self._interactions.T

    def conditional_probabilities(self, responses: NDArray[np.int_]) -> FloatArray:
        """Return every node's full-conditional success probability."""

        values = _validate_binary_responses(responses, n_nodes=self._n_nodes)
        return _stable_sigmoid(self._linear_predictors(values))

    def conditional_probability(
        self,
        node_idx: int,
        other_values: NDArray[np.int_],
    ) -> FloatArray:
        """Compute ``P(X_i=1 | X_-i)`` for one node.

        ``other_values`` may contain all nodes or only the ``n_nodes - 1``
        conditioning nodes in their original order. A full row's value for
        ``node_idx`` is ignored, and a single row may be given as a vector.
        """

        node_idx = _validate_node_index(node_idx, self._n_nodes)
        values = np.asarray(other_values)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim == 2 and values.shape[1] == self._n_nodes - 1:
            # The node's own value has no weight in its linear predictor.
            values = np.insert(values, node_idx, 0, axis=1)
        elif values.ndim != 2 or values.shape[1] != self._n_nodes:
            raise MirtDataError(
                f"other_values must have {self._n_nodes} or {self._n_nodes - 1} columns"
            )
        return self.conditional_probabilities(values)[:, node_idx]

    def pseudo_likelihood(self, responses: NDArray[np.int_]) -> float:
        """Compute mean log-pseudolikelihood across observations."""

        values = _validate_binary_responses(responses, n_nodes=self._n_nodes)
        linear = self._linear_predictors(values)
        log_terms = values * linear - np.logaddexp(0.0, linear)
        return float(np.sum(log_terms) / values.shape[0])

    def _validate_exact_enumeration(self, max_nodes: int) -> None:
        """Reject exact enumeration beyond the caller's explicit limit."""

        if self._n_nodes > max_nodes:
            raise MirtValidationError(
                f"Exact enumeration requires {2**self._n_nodes:,} states",
                parameter="max_nodes",
                value=max_nodes,
                expected=f">= {self._n_nodes}",
            )

    def _exact_state_energies(self) -> FloatArray:
        """Return unnormalized log probabilities for every binary state."""

        state_ids = np.arange(2**self._n_nodes, dtype=np.uint64)[:, None]
        bit_positions = np.arange(self._n_nodes, dtype=np.uint64)
        states = ((state_ids >> bit_positions) & 1).astype(np.float64)
        return states @ self._thresholds + 0.5 * np.einsum(
            "bi,ij,bj->b", states, self._interactions, states, optimize=True
        )

    def _exact_state_cdf(self) -> FloatArray:
        """Return a cached cumulative distribution over binary state IDs."""

        if self._exact_cdf_cache is None:
            energies = self._exact_state_energies()
            if self._log_partition_cache is None:
                self._log_partition_cache = float(np.logaddexp.reduce(energies))
            probabilities = np.exp(energies - self._log_partition_cache)
            cumulative = np.cumsum(probabilities, dtype=np.float64)
            cumulative /= cumulative[-1]
            cumulative[-1] = 1.0
            cumulative.setflags(write=False)
            self._exact_cdf_cache = cumulative
        return self._exact_cdf_cache

    def log_partition_function(self, max_nodes: int = 16) -> float:
        """Compute the exact log partition function for a small network.

        Exact enumeration grows exponentially, so the default guard rejects
        networks larger than 16 nodes. Increase ``max_nodes`` explicitly when
        the memory and runtime cost is acceptable.
        """

        max_nodes = _validate_count("max_nodes", max_nodes, allow_zero=False)
        self._validate_exact_enumeration(max_nodes)
        if self._log_partition_cache is not None:
            return self._log_partition_cache

        energies = self._exact_state_energies()
        self._log_partition_cache = float(np.logaddexp.reduce(energies))
        return self._log_partition_cache

    def log_probability(
        self,
        responses: NDArray[np.int_],
        *,
        max_nodes: int = 16,
    ) -> FloatArray:
        """Compute exact normalized log probabilities for small networks."""

        values = _validate_binary_responses(responses, n_nodes=self._n_nodes)
        energies = values @ self._thresholds + 0.5 * np.einsum(
            "bi,ij,bj->b", values, self._interactions, values, optimize=True
        )
        return energies - self.log_partition_function(max_nodes=max_nodes)

    def probability(
        self,
        responses: NDArray[np.int_],
        *,
        max_nodes: int = 16,
    ) -> FloatArray:
        """Compute exact normalized probabilities for small networks."""

        return np.exp(self.log_probability(responses, max_nodes=max_nodes))

    def sample(
        self,
        n_samples: int,
        n_burnin: int = 1000,
        seed: int | None = None,
        thin: int = 1,
        *,
        method: Literal["gibbs", "exact"] = "gibbs",
        max_nodes: int = 16,
    ) -> IntArray:
        """Generate binary network samples.

        Gibbs sampling supports burn-in and thinning for networks of any size.
        Exact sampling draws independent states and is guarded by ``max_nodes``
        because its setup cost doubles with every node.
        """

        n_samples = _validate_count("n_samples", n_samples, allow_zero=True)
        n_burnin = _validate_count("n_burnin", n_burnin, allow_zero=True)
        thin = _validate_count("thin", thin, allow_zero=False)
        max_nodes = _validate_count("max_nodes", max_nodes, allow_zero=False)
        if method not in {"gibbs", "exact"}:
            raise MirtValidationError(
                "method must be 'gibbs' or 'exact'",
                parameter="method",
                value=method,
            )
        if method == "exact":
            self._validate_exact_enumeration(max_nodes)

        samples = np.zeros((n_samples, self._n_nodes), dtype=np.int64)
        if n_samples == 0:
            return samples

        rng = np.random.default_rng(seed)
        if method == "exact":
            state_ids = np.searchsorted(
                self._exact_state_cdf(),
                rng.random(n_samples),
                side="right",
            ).astype(np.uint64, copy=False)
            bit_positions = np.arange(self._n_nodes, dtype=np.uint64)
            return ((state_ids[:, None] >> bit_positions) & 1).astype(np.int64)

        current = rng.binomial(1, 0.5, self._n_nodes).astype(np.int64)
        linear = self._thresholds + self._interactions @ current
        sample_idx = 0
        total_sweeps = n_burnin + n_samples * thin
        for sweep in range(total_sweeps):
            for node in range(self._n_nodes):
                node_linear = linear[node]
                if node_linear >= 0.0:
                    probability = 1.0 / (1.0 + np.exp(-node_linear))
                else:
                    exponential = np.exp(node_linear)
                    probability = exponential / (1.0 + exponential)
                updated = int(rng.random() < probability)
                change = updated - current[node]
                if change:
                    current[node] = updated
                    linear += self._interactions[:, node] * change
            if sweep >= n_burnin and (sweep - n_burnin) % thin == 0:
                samples[sample_idx] = current
                sample_idx += 1
        return samples

    def edge_weights(self) -> FloatArray:
        """Return the interaction matrix's upper triangle."""

        return np.triu(self._interactions, k=1)

    def degree_centrality(self) -> FloatArray:
        """Return normalized absolute node strength."""

        return np.sum(np.abs(self._interactions), axis=1) / (self._n_nodes - 1)

    def expected_influence(self) -> FloatArray:
        """Return signed node strength."""

        return np.sum(self._interactions, axis=1)

    def copy(self) -> Self:
        copied = self.__class__(self._n_nodes, self._node_names)
        copied._thresholds = self._thresholds.copy()
        copied._interactions = self._interactions.copy()
        copied._is_fitted = self._is_fitted
        copied._n_iterations = self._n_iterations
        copied._converged = self._converged
        copied._objective_history = self._objective_history.copy()
        copied._log_partition_cache = self._log_partition_cache
        if self._exact_cdf_cache is not None:
            copied._exact_cdf_cache = self._exact_cdf_cache.copy()
            copied._exact_cdf_cache.setflags(write=False)
        return copied


class GaussianGraphicalModel:
    """Gaussian network represented by a positive-definite precision matrix."""

    def __init__(
        self,
        n_nodes: int,
        node_names: list[str] | None = None,
    ) -> None:
        self._n_nodes, self._node_names = _validate_node_metadata(n_nodes, node_names)
        self._means = np.zeros(self._n_nodes, dtype=np.float64)
        self._precision = np.eye(self._n_nodes, dtype=np.float64)
        self._is_fitted = False
        self._n_iterations = 0
        self._converged = False
        self._objective_history: list[float] = []
        self._covariance_cache: FloatArray | None = None
        self._log_precision_determinant = 0.0

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def node_names(self) -> list[str]:
        return self._node_names.copy()

    @property
    def means(self) -> FloatArray:
        return self._means.copy()

    @property
    def precision_matrix(self) -> FloatArray:
        return self._precision.copy()

    @property
    def covariance_matrix(self) -> FloatArray:
        """Return the inverse precision matrix."""

        if self._covariance_cache is None:
            self._covariance_cache = np.linalg.solve(
                self._precision,
                np.eye(self._n_nodes),
            )
        return self._covariance_cache.copy()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_iterations(self) -> int:
        return self._n_iterations

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def objective_history(self) -> FloatArray:
        return np.asarray(self._objective_history, dtype=np.float64)

    def set_means(self, means: NDArray[np.float64]) -> Self:
        values = np.asarray(means, dtype=np.float64)
        if values.shape != (self._n_nodes,):
            raise MirtValidationError(
                f"means shape {values.shape} != ({self._n_nodes},)",
                parameter="means",
            )
        if not np.all(np.isfinite(values)):
            raise MirtValidationError(
                "means must contain only finite values", parameter="means"
            )
        self._means = values.copy()
        return self

    def set_precision_matrix(self, precision: NDArray[np.float64]) -> Self:
        values = np.asarray(precision, dtype=np.float64)
        expected = (self._n_nodes, self._n_nodes)
        if values.shape != expected:
            raise MirtValidationError(
                f"precision shape {values.shape} != {expected}",
                parameter="precision",
            )
        if not np.all(np.isfinite(values)):
            raise MirtValidationError(
                "precision must contain only finite values", parameter="precision"
            )
        values = (values + values.T) / 2.0
        try:
            cholesky = np.linalg.cholesky(values)
        except np.linalg.LinAlgError as error:
            raise MirtValidationError(
                "precision matrix must be positive definite",
                parameter="precision",
            ) from error
        self._precision = values
        self._covariance_cache = None
        self._log_precision_determinant = float(2.0 * np.log(np.diag(cholesky)).sum())
        return self

    def partial_correlations(self) -> FloatArray:
        """Return standardized negative off-diagonal precision entries."""

        scale = np.sqrt(np.diag(self._precision))
        partial = -self._precision / np.outer(scale, scale)
        np.fill_diagonal(partial, 1.0)
        return partial

    def conditional_mean(
        self,
        node_idx: int,
        other_values: NDArray[np.float64],
    ) -> FloatArray:
        """Compute a node's conditional mean.

        ``other_values`` may contain all nodes or only the ``n_nodes - 1``
        conditioning nodes in their original order.
        """

        node_idx = _validate_node_index(node_idx, self._n_nodes)
        values = np.asarray(other_values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] not in (
            self._n_nodes,
            self._n_nodes - 1,
        ):
            raise MirtDataError(
                f"other_values must have {self._n_nodes} or {self._n_nodes - 1} columns"
            )
        if not np.all(np.isfinite(values)):
            raise MirtDataError("other_values must contain only finite values")

        other_means = np.delete(self._means, node_idx)
        if values.shape[1] == self._n_nodes:
            centered = np.delete(values, node_idx, axis=1) - other_means
        else:
            centered = values - other_means
        precision_row = np.delete(self._precision[node_idx], node_idx)
        return (
            self._means[node_idx]
            - centered @ precision_row / self._precision[node_idx, node_idx]
        )

    def conditional_variance(self, node_idx: int) -> float:
        """Compute a node's conditional variance."""

        node_idx = _validate_node_index(node_idx, self._n_nodes)
        return float(1.0 / self._precision[node_idx, node_idx])

    def log_likelihood(self, data: NDArray[np.float64]) -> float:
        """Compute the Gaussian log likelihood."""

        values = _validate_continuous_data(data, n_nodes=self._n_nodes)
        centered = values - self._means
        quadratic = np.einsum(
            "bi,ij,bj->", centered, self._precision, centered, optimize=True
        )
        return float(
            -0.5 * values.shape[0] * self._n_nodes * np.log(2.0 * np.pi)
            + 0.5 * values.shape[0] * self._log_precision_determinant
            - 0.5 * quadratic
        )

    def sample(
        self,
        n_samples: int,
        seed: int | None = None,
    ) -> FloatArray:
        """Generate multivariate normal samples."""

        n_samples = _validate_count("n_samples", n_samples, allow_zero=True)
        if n_samples == 0:
            return np.empty((0, self._n_nodes), dtype=np.float64)
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(
            self._means, self.covariance_matrix, size=n_samples
        )

    def edge_weights(self) -> FloatArray:
        """Return upper-triangular partial correlations."""

        return np.triu(self.partial_correlations(), k=1)

    def degree_centrality(self) -> FloatArray:
        """Return normalized absolute node strength."""

        partial = self.partial_correlations()
        np.fill_diagonal(partial, 0.0)
        return np.sum(np.abs(partial), axis=1) / (self._n_nodes - 1)

    def expected_influence(self) -> FloatArray:
        """Return signed partial-correlation strength."""

        partial = self.partial_correlations()
        np.fill_diagonal(partial, 0.0)
        return np.sum(partial, axis=1)

    def copy(self) -> Self:
        copied = self.__class__(self._n_nodes, self._node_names)
        copied._means = self._means.copy()
        copied._precision = self._precision.copy()
        copied._is_fitted = self._is_fitted
        copied._n_iterations = self._n_iterations
        copied._converged = self._converged
        copied._objective_history = self._objective_history.copy()
        copied._covariance_cache = (
            None if self._covariance_cache is None else self._covariance_cache.copy()
        )
        copied._log_precision_determinant = self._log_precision_determinant
        return copied


def _ising_objective(
    responses: IntArray,
    thresholds: FloatArray,
    interactions: FloatArray,
    regularization: float,
) -> float:
    linear = thresholds + responses @ interactions.T
    log_pseudo_likelihood = (
        np.sum(responses * linear - np.logaddexp(0.0, linear)) / responses.shape[0]
    )
    penalty = regularization * np.sum(np.abs(np.triu(interactions, k=1)))
    return float(log_pseudo_likelihood - penalty)


def _ising_pseudo_likelihood_terms(
    responses: FloatArray,
    thresholds: FloatArray,
    interactions: FloatArray,
    upper: tuple[IntArray, IntArray],
) -> tuple[float, FloatArray, FloatArray]:
    """Return the mean log-pseudolikelihood and its threshold and edge gradients.

    ``interactions`` must be symmetric with a zero diagonal, and ``upper``
    indexes its unique edges.
    """
    n_samples = responses.shape[0]
    linear = thresholds + responses @ interactions
    log_pseudo_likelihood = float(
        np.sum(responses * linear - np.logaddexp(0.0, linear)) / n_samples
    )
    residual = responses - _stable_sigmoid(linear)
    node_gradient = residual.T @ responses / n_samples
    # Edge (i, j) enters the full conditionals of both node i and node j.
    edge_gradient = (node_gradient + node_gradient.T)[upper]
    return log_pseudo_likelihood, residual.mean(axis=0), edge_gradient


def _ising_optimality_residual(
    threshold_gradient: FloatArray,
    edge_gradient: FloatArray,
    edges: FloatArray,
    regularization: float,
) -> float:
    """Return the largest minimum-norm subgradient of the penalized objective.

    It is zero exactly at the penalized pseudolikelihood maximum: thresholds
    and nonzero edges have zero gradient, and every zero edge's gradient lies
    within ``[-regularization, regularization]``.
    """
    edge_residual = np.where(
        edges != 0.0,
        np.abs(edge_gradient - regularization * np.sign(edges)),
        np.maximum(np.abs(edge_gradient) - regularization, 0.0),
    )
    return float(
        max(np.max(np.abs(threshold_gradient)), np.max(edge_residual, initial=0.0))
    )


def fit_ising(
    responses: NDArray[np.int_],
    regularization: float = 0.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
    node_names: list[str] | None = None,
) -> tuple[IsingModel, float]:
    """Fit a symmetric Ising network by penalized pseudolikelihood.

    The thresholds and unique symmetric edges jointly maximize the mean
    log-pseudolikelihood minus ``regularization`` times the sum of absolute
    edge weights; thresholds are not penalized. L-BFGS-B solves the problem
    with analytic gradients. With a penalty, each edge is split into
    nonnegative positive and negative parts, which makes the L1 term linear;
    an edge whose parts both stop at their zero bound is exactly zero.

    Parameters
    ----------
    responses : NDArray
        Binary ``(n_observations, n_nodes)`` matrix with at least two rows.
    regularization : float
        Non-negative L1 penalty on the edge weights.
    max_iter : int
        Maximum number of L-BFGS-B iterations.
    tol : float
        Convergence tolerance on the optimality residual: the largest
        absolute gradient of the penalized objective with respect to the
        thresholds and nonzero edges, and the largest amount by which a zero
        edge's gradient exceeds ``regularization`` in magnitude.
    verbose : bool
        Print the penalized objective after every iteration.
    node_names : list of str, optional
        Node names.

    Returns
    -------
    tuple
        The fitted :class:`IsingModel` and its mean log-pseudolikelihood.
        ``model.converged`` reports whether the optimality residual reached
        ``tol``, and ``model.objective_history`` holds the penalized objective
        at the starting values and after every iteration.

    Warns
    -----
    RuntimeWarning
        If the optimality residual still exceeds ``tol`` when the solver
        stops, for example after ``max_iter`` iterations or when a node or
        node pair has no finite pseudolikelihood maximum.
    """

    values = _validate_binary_responses(responses, minimum_observations=2)
    regularization, max_iter, tol = _validate_optimizer_options(
        regularization, max_iter, tol
    )
    n_nodes = values.shape[1]
    model = IsingModel(n_nodes, node_names)
    observations = values.astype(np.float64)
    upper = np.triu_indices(n_nodes, k=1)
    n_edges = upper[0].size
    penalized = regularization > 0.0

    def unpack(
        parameters: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        edges = parameters[n_nodes : n_nodes + n_edges]
        if penalized:
            edges = edges - parameters[n_nodes + n_edges :]
        interactions = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        interactions[upper] = edges
        return parameters[:n_nodes], interactions + interactions.T, edges

    def negative_objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        thresholds, interactions, _ = unpack(parameters)
        log_pseudo_likelihood, threshold_gradient, edge_gradient = (
            _ising_pseudo_likelihood_terms(
                observations, thresholds, interactions, upper
            )
        )
        if not penalized:
            return -log_pseudo_likelihood, -np.concatenate(
                (threshold_gradient, edge_gradient)
            )
        # With edges = positive - negative parts, both parts >= 0, the penalty
        # is linear and the solver only needs simple bounds.
        penalty = regularization * float(np.sum(parameters[n_nodes:]))
        return -log_pseudo_likelihood + penalty, np.concatenate(
            (
                -threshold_gradient,
                regularization - edge_gradient,
                regularization + edge_gradient,
            )
        )

    probabilities = np.clip(values.mean(axis=0), 1e-6, 1.0 - 1e-6)
    initial = np.zeros(n_nodes + (2 if penalized else 1) * n_edges)
    initial[:n_nodes] = np.log(probabilities / (1.0 - probabilities))
    history = [-negative_objective(initial)[0]]

    def record(intermediate_result: OptimizeResult) -> None:
        history.append(-float(intermediate_result.fun))
        if verbose:
            print(f"Iteration {len(history) - 1}: objective = {history[-1]:.6f}")

    bounds = (
        [(None, None)] * n_nodes + [(0.0, None)] * (2 * n_edges) if penalized else None
    )
    result = minimize(
        negative_objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        callback=record,
        options={
            "maxiter": max_iter,
            "maxfun": max(15_000, 20 * max_iter),
            "gtol": tol,
            # Stop on the optimality residual rather than on slow progress.
            "ftol": np.finfo(np.float64).eps,
        },
    )

    thresholds, interactions, edges = unpack(np.asarray(result.x, dtype=np.float64))
    _, threshold_gradient, edge_gradient = _ising_pseudo_likelihood_terms(
        observations, thresholds, interactions, upper
    )
    residual = _ising_optimality_residual(
        threshold_gradient, edge_gradient, edges, regularization
    )
    converged = residual <= tol
    if not converged:
        warnings.warn(
            f"fit_ising stopped after {result.nit} iterations with optimality "
            f"residual {residual:.2e} above tol={tol:.2e}; increase max_iter, "
            "or add regularization if a node or node pair is perfectly "
            "predictable",
            RuntimeWarning,
            stacklevel=2,
        )

    model.set_thresholds(thresholds)
    model.set_interactions(interactions)
    model._is_fitted = True
    model._n_iterations = int(result.nit)
    model._converged = converged
    model._objective_history = history
    return model, model.pseudo_likelihood(values)


def _ggm_objective(
    sample_covariance: FloatArray,
    precision: FloatArray,
    regularization: float,
) -> float:
    sign, log_determinant = np.linalg.slogdet(precision)
    if sign <= 0:
        return -np.inf
    off_diagonal = precision.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    return float(
        log_determinant
        - np.sum(sample_covariance * precision)
        - regularization * np.sum(np.abs(off_diagonal))
    )


def fit_ggm(
    data: NDArray[np.float64],
    regularization: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
    node_names: list[str] | None = None,
) -> tuple[GaussianGraphicalModel, float]:
    """Fit a Gaussian graphical model with an off-diagonal L1 penalty.

    Regularized fits use an adaptive ADMM solver whose precision update is
    positive definite by construction. When ``regularization`` is zero, the
    sample covariance must be invertible.
    """

    values = _validate_continuous_data(data, minimum_observations=2)
    regularization, max_iter, tol = _validate_optimizer_options(
        regularization, max_iter, tol
    )
    _, n_nodes = values.shape
    model = GaussianGraphicalModel(n_nodes, node_names)

    means = values.mean(axis=0)
    centered = values - means
    sample_covariance = centered.T @ centered / values.shape[0]
    scale = max(float(np.trace(sample_covariance) / n_nodes), 1.0)
    eigenvalue_floor = np.finfo(np.float64).eps * scale * n_nodes

    if regularization == 0.0:
        if np.min(np.linalg.eigvalsh(sample_covariance)) <= eigenvalue_floor:
            raise MirtValidationError(
                "sample covariance is singular; use regularization > 0",
                parameter="regularization",
                value=regularization,
            )
        precision = np.linalg.solve(sample_covariance, np.eye(n_nodes))
        history = [_ggm_objective(sample_covariance, precision, 0.0)]
        converged = True
        n_iterations = 0
    else:
        regularized_covariance = sample_covariance + regularization * np.eye(n_nodes)
        precision = np.linalg.solve(regularized_covariance, np.eye(n_nodes))
        sparse_precision = precision.copy()
        scaled_dual = np.zeros_like(precision)
        penalty_scale = 1.0
        history = [_ggm_objective(sample_covariance, precision, regularization)]
        converged = False

        for iteration in range(1, max_iter + 1):
            update_matrix = (
                penalty_scale * (sparse_precision - scaled_dual) - sample_covariance
            )
            eigenvalues, eigenvectors = np.linalg.eigh(update_matrix)
            updated_eigenvalues = (
                eigenvalues + np.sqrt(eigenvalues**2 + 4.0 * penalty_scale)
            ) / (2.0 * penalty_scale)
            precision = (eigenvectors * updated_eigenvalues) @ eigenvectors.T

            previous_sparse = sparse_precision.copy()
            threshold_input = precision + scaled_dual
            sparse_precision = _soft_threshold(
                threshold_input, regularization / penalty_scale
            )
            np.fill_diagonal(sparse_precision, np.diag(threshold_input))
            sparse_precision = (sparse_precision + sparse_precision.T) / 2.0
            scaled_dual += precision - sparse_precision

            primal_residual = float(np.linalg.norm(precision - sparse_precision))
            dual_residual = float(
                penalty_scale * np.linalg.norm(sparse_precision - previous_sparse)
            )
            primal_tolerance = n_nodes * tol + tol * max(
                float(np.linalg.norm(precision)),
                float(np.linalg.norm(sparse_precision)),
            )
            dual_tolerance = n_nodes * tol + tol * penalty_scale * float(
                np.linalg.norm(scaled_dual)
            )
            history.append(_ggm_objective(sample_covariance, precision, regularization))

            if verbose:
                print(
                    f"Iteration {iteration}: objective = {history[-1]:.6f}, "
                    f"primal = {primal_residual:.3e}, dual = {dual_residual:.3e}"
                )
            if primal_residual <= primal_tolerance and dual_residual <= dual_tolerance:
                converged = True
                break

            if primal_residual > 10.0 * dual_residual:
                penalty_scale *= 2.0
                scaled_dual /= 2.0
            elif dual_residual > 10.0 * primal_residual:
                penalty_scale /= 2.0
                scaled_dual *= 2.0
        n_iterations = len(history) - 1

        if converged:
            try:
                np.linalg.cholesky(sparse_precision)
            except np.linalg.LinAlgError:
                pass
            else:
                precision = sparse_precision

    model.set_means(means)
    model.set_precision_matrix(precision)
    model._is_fitted = True
    model._n_iterations = n_iterations
    model._converged = converged
    model._objective_history = history
    return model, model.log_likelihood(values)


def _network_weight_matrix(
    model: IsingModel | GaussianGraphicalModel,
) -> FloatArray:
    if isinstance(model, IsingModel):
        return model.interactions
    partial = model.partial_correlations()
    np.fill_diagonal(partial, 0.0)
    return partial


def compare_networks(
    model1: IsingModel | GaussianGraphicalModel,
    model2: IsingModel | GaussianGraphicalModel,
    *,
    edge_threshold: float = 1e-8,
) -> dict[str, float]:
    """Compare corresponding edges and node strengths in two networks."""

    if not isinstance(model1, type(model2)):
        raise MirtValidationError("Models must be of the same type")
    if model1.n_nodes != model2.n_nodes:
        raise MirtValidationError("Models must have the same number of nodes")
    edge_threshold = float(edge_threshold)
    if not np.isfinite(edge_threshold) or edge_threshold < 0.0:
        raise MirtValidationError(
            "edge_threshold must be finite and non-negative",
            parameter="edge_threshold",
            value=edge_threshold,
        )
    if set(model1.node_names) != set(model2.node_names):
        raise MirtValidationError("Models must contain the same node names")

    model2_order = np.asarray(
        [model2.node_names.index(name) for name in model1.node_names]
    )
    weights1 = _network_weight_matrix(model1)
    weights2 = _network_weight_matrix(model2)[np.ix_(model2_order, model2_order)]
    upper = np.triu_indices(model1.n_nodes, k=1)
    edges1 = weights1[upper]
    edges2 = weights2[upper]

    degree1 = np.sum(np.abs(weights1), axis=1) / (model1.n_nodes - 1)
    degree2 = np.sum(np.abs(weights2), axis=1) / (model2.n_nodes - 1)
    edge_difference = np.abs(edges1 - edges2)
    present1 = np.abs(edges1) > edge_threshold
    present2 = np.abs(edges2) > edge_threshold
    union = np.count_nonzero(present1 | present2)
    intersection = np.count_nonzero(present1 & present2)

    def safe_correlation(left: FloatArray, right: FloatArray) -> float:
        if left.size < 2:
            return np.nan
        scale = np.finfo(np.float64).eps
        if np.std(left) <= scale or np.std(right) <= scale:
            return np.nan
        return float(np.corrcoef(left, right)[0, 1])

    return {
        "edge_correlation": safe_correlation(edges1, edges2),
        "degree_correlation": safe_correlation(degree1, degree2),
        "mean_edge_difference": float(np.mean(edge_difference)),
        "max_edge_difference": float(np.max(edge_difference)),
        "frobenius_difference": float(np.linalg.norm(weights1 - weights2)),
        "edge_jaccard": float(intersection / union) if union else 1.0,
    }
