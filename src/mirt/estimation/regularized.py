from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON, REGULARIZATION_EPSILON
from mirt.estimation.base import BaseEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.exceptions import MirtDataError
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.estimation.latent_density import LatentDensity
    from mirt.models.multidimensional import MultidimensionalModel

_PENALTY_TYPES = frozenset(("lasso", "ridge", "elastic_net"))
_MAX_ESTEP_TEMP_ENTRIES = 1_000_000


def _validate_integer(value: object, name: str, *, minimum: int) -> int:
    """Return a validated integer configuration value."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return int(value)


def _validate_real(
    value: object,
    name: str,
    *,
    minimum: float,
    inclusive: bool,
) -> float:
    """Return a validated finite real configuration value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        qualifier = "greater than or equal to" if inclusive else "greater than"
        raise ValueError(f"{name} must be a finite real number {qualifier} {minimum}")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        qualifier = "greater than or equal to" if inclusive else "greater than"
        raise ValueError(
            f"{name} must be a finite real number {qualifier} {minimum}"
        ) from error
    valid_bound = normalized >= minimum if inclusive else normalized > minimum
    if not np.isfinite(normalized) or not valid_bound:
        qualifier = "greater than or equal to" if inclusive else "greater than"
        raise ValueError(f"{name} must be a finite real number {qualifier} {minimum}")
    return normalized


def _binary_response_components(
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return correct and observed indicator matrices for binary responses."""
    correct = (responses == 1).astype(np.float64)
    observed = (responses >= 0).astype(np.float64)
    return correct, observed


def _expected_item_counts(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected correct and observed counts for all items."""
    correct, observed = _binary_response_components(responses)
    return correct.T @ posterior_weights, observed.T @ posterior_weights


@dataclass
class PenaltySpec:
    """Specification for regularization penalty."""

    type: Literal["lasso", "ridge", "elastic_net"]
    lambda_val: float
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or self.type not in _PENALTY_TYPES:
            choices = ", ".join(sorted(_PENALTY_TYPES))
            raise ValueError(f"penalty type must be one of: {choices}")
        self.lambda_val = _validate_real(
            self.lambda_val,
            "lambda_val",
            minimum=0.0,
            inclusive=True,
        )
        self.alpha = _validate_real(
            self.alpha,
            "alpha",
            minimum=0.0,
            inclusive=True,
        )
        if self.alpha > 1.0:
            raise ValueError("alpha must be between 0 and 1")


@dataclass
class RegularizedMIRTResult:
    """Result from regularized MIRT estimation."""

    model: MultidimensionalModel
    loadings: NDArray[np.float64]
    intercepts: NDArray[np.float64]
    lambda_val: float
    alpha: float
    penalty_type: str
    log_likelihood: float
    penalized_ll: float
    n_nonzero: int
    aic: float
    bic: float
    ebic: float
    converged: bool
    n_iterations: int
    n_observations: int
    n_parameters: int

    def summary(self) -> str:
        lines = []
        width = 80

        lines.append("=" * width)
        lines.append(f"{'Regularized MIRT Results':^{width}}")
        lines.append("=" * width)

        lines.append(
            f"Penalty Type:       {self.penalty_type:<20} Lambda:            {self.lambda_val:>12.6f}"
        )
        lines.append(
            f"Alpha (EN mixing):  {self.alpha:<20.4f} Non-zero loadings: {self.n_nonzero:>12}"
        )
        lines.append(
            f"Log-Likelihood:     {self.log_likelihood:<20.4f} Penalized LL:      {self.penalized_ll:>12.4f}"
        )
        lines.append(
            f"AIC:                {self.aic:<20.4f} BIC:               {self.bic:>12.4f}"
        )
        lines.append(
            f"EBIC:               {self.ebic:<20.4f} Converged:         {str(self.converged):>12}"
        )
        lines.append("-" * width)

        lines.append("\nFactor Loadings:")
        n_items, n_factors = self.loadings.shape

        header = f"{'Item':<15}"
        for f in range(n_factors):
            header += f"{'F' + str(f + 1):>10}"
        header += f"{'Intercept':>12}"
        lines.append(header)
        lines.append("-" * width)

        item_names = (
            self.model.item_names
            if hasattr(self.model, "item_names")
            else [f"Item_{i}" for i in range(n_items)]
        )

        for i in range(n_items):
            row = f"{item_names[i]:<15}"
            for f in range(n_factors):
                val = self.loadings[i, f]
                if abs(val) < REGULARIZATION_EPSILON:
                    row += f"{'---':>10}"
                else:
                    row += f"{val:>10.4f}"
            row += f"{self.intercepts[i]:>12.4f}"
            lines.append(row)

        lines.append("=" * width)
        return "\n".join(lines)

    def simple_structure(self, threshold: float = 0.3) -> NDArray[np.bool_]:
        """Return boolean mask of loadings above threshold."""
        return np.abs(self.loadings) >= threshold

    def loading_pattern(self, threshold: float = 0.3) -> NDArray[np.int_]:
        """Return pattern matrix (1 for significant loadings, 0 otherwise)."""
        return (np.abs(self.loadings) >= threshold).astype(np.int_)


class RegularizedMIRTEstimator(BaseEstimator):
    """Penalized estimation for exploratory MIRT with sparse factor loadings.

    This estimator applies LASSO, ridge, or elastic net regularization to the
    factor loading matrix during EM estimation, enabling automatic discovery
    of simple structure.

    Parameters
    ----------
    penalty : PenaltySpec or str
        Regularization penalty specification. Can be "lasso", "ridge",
        "elastic_net", or a PenaltySpec object.
    lambda_val : float, optional
        Regularization strength. If None, must be provided at fit time.
    alpha : float
        Elastic net mixing parameter (1 = pure LASSO, 0 = pure ridge).
    n_factors : int
        Number of latent factors to estimate.
    n_quadpts : int
        Number of quadrature points per dimension.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.
    cd_max_iter : int
        Maximum coordinate descent iterations per M-step.
    cd_tol : float
        Coordinate descent convergence tolerance.
    adaptive : bool
        Use adaptive LASSO with weights from initial unpenalized fit.
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        penalty: PenaltySpec | Literal["lasso", "ridge", "elastic_net"] = "lasso",
        lambda_val: float | None = None,
        alpha: float = 1.0,
        n_factors: int = 2,
        n_quadpts: int = 15,
        max_iter: int = 500,
        tol: float = 1e-4,
        cd_max_iter: int = 100,
        cd_tol: float = 1e-6,
        adaptive: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        if isinstance(penalty, str):
            if lambda_val is None:
                lambda_val = 0.1
            self.penalty = PenaltySpec(type=penalty, lambda_val=lambda_val, alpha=alpha)
        elif isinstance(penalty, PenaltySpec):
            self.penalty = penalty
        else:
            raise TypeError("penalty must be a penalty name or PenaltySpec")

        self.n_factors = _validate_integer(n_factors, "n_factors", minimum=2)
        self.n_quadpts = _validate_integer(n_quadpts, "n_quadpts", minimum=1)
        self.cd_max_iter = _validate_integer(cd_max_iter, "cd_max_iter", minimum=1)
        self.cd_tol = _validate_real(
            cd_tol,
            "cd_tol",
            minimum=0.0,
            inclusive=False,
        )
        if not isinstance(adaptive, (bool, np.bool_)):
            raise TypeError("adaptive must be a boolean")
        self.adaptive = bool(adaptive)
        self._quadrature: GaussHermiteQuadrature | None = None
        self._adaptive_weights: NDArray[np.float64] | None = None

    def _validated_binary_responses(
        self,
        responses: NDArray[np.int_],
    ) -> NDArray[np.int_]:
        """Validate a non-empty binary response matrix."""
        response_array = np.asarray(responses)
        if response_array.ndim != 2:
            raise ValueError(f"responses must be 2D, got {response_array.ndim}D")
        n_items = response_array.shape[1]
        validated = self._validate_responses(response_array, n_items)
        if np.any(validated > 1):
            raise MirtDataError(
                "regularized MIRT requires binary responses coded as 0 or 1",
                n_persons=validated.shape[0],
                n_items=n_items,
            )
        return validated

    def fit(
        self,
        responses: NDArray[np.int_],
        lambda_val: float | None = None,
    ) -> RegularizedMIRTResult:
        """Fit regularized MIRT model.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items). Missing values coded as -1.
        lambda_val : float, optional
            Override regularization strength.

        Returns
        -------
        RegularizedMIRTResult
            Fitted model with sparse loadings.
        """
        from mirt.estimation.latent_density import GaussianDensity
        from mirt.models.multidimensional import MultidimensionalModel

        responses = self._validated_binary_responses(responses)
        n_persons, n_items = responses.shape

        if lambda_val is not None:
            self.penalty.lambda_val = _validate_real(
                lambda_val,
                "lambda_val",
                minimum=0.0,
                inclusive=True,
            )

        model = MultidimensionalModel(
            n_items=n_items,
            n_factors=self.n_factors,
            model_type="exploratory",
        )
        model._initialize_parameters()

        self._quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=self.n_factors,
        )

        latent_density = GaussianDensity(
            mean=np.zeros(self.n_factors),
            cov=np.eye(self.n_factors),
            n_dimensions=self.n_factors,
        )

        loadings = model.slopes.copy()
        intercepts = model.intercepts.copy()

        if self.adaptive:
            init_loadings = self._fit_unpenalized_warmstart(
                responses,
                loadings.copy(),
                intercepts.copy(),
                latent_density,
                max_iter=50,
            )
            self._adaptive_weights = 1.0 / (
                np.abs(init_loadings) + REGULARIZATION_EPSILON
            )
        else:
            self._adaptive_weights = np.ones_like(loadings)

        valid_masks = [responses[:, j] >= 0 for j in range(n_items)]

        self._convergence_history = []
        prev_ll = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            posterior_weights, marginal_ll = self._e_step(
                responses, loadings, intercepts, latent_density
            )

            current_ll = np.sum(np.log(marginal_ll + 1e-300))
            penalty_term = self._compute_penalty(loadings)
            penalized_ll = current_ll - penalty_term

            self._convergence_history.append(current_ll)
            self._log_iteration(iteration, current_ll, penalty=penalty_term)

            if self._check_convergence(prev_ll, current_ll):
                converged = True
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_ll = current_ll

            loadings, intercepts = self._m_step_penalized(
                responses, posterior_weights, loadings, intercepts, valid_masks
            )

            n_k = posterior_weights.sum(axis=0)
            latent_density.update(self._quadrature.nodes, n_k)
        else:
            posterior_weights, marginal_ll = self._e_step(
                responses, loadings, intercepts, latent_density
            )
            current_ll = float(np.sum(np.log(marginal_ll + 1e-300)))
            penalty_term = self._compute_penalty(loadings)
            penalized_ll = current_ll - penalty_term
            self._convergence_history.append(current_ll)
            converged = self._check_convergence(prev_ll, current_ll)

        model.set_parameters(slopes=loadings, intercepts=intercepts)
        model._is_fitted = True

        n_nonzero = int(np.sum(np.abs(loadings) > REGULARIZATION_EPSILON))
        n_params = n_nonzero + n_items

        aic = -2 * current_ll + 2 * n_params
        bic = -2 * current_ll + n_params * np.log(n_persons)
        gamma = 0.5
        ebic = bic + 2 * gamma * np.log(
            self._count_possible_models(n_items, self.n_factors)
        )

        return RegularizedMIRTResult(
            model=model,
            loadings=loadings,
            intercepts=intercepts,
            lambda_val=self.penalty.lambda_val,
            alpha=self.penalty.alpha,
            penalty_type=self.penalty.type,
            log_likelihood=current_ll,
            penalized_ll=penalized_ll,
            n_nonzero=n_nonzero,
            aic=aic,
            bic=bic,
            ebic=ebic,
            converged=converged,
            n_iterations=iteration + 1,
            n_observations=n_persons,
            n_parameters=n_params,
        )

    def fit_path(
        self,
        responses: NDArray[np.int_],
        lambda_values: list[float] | None = None,
        n_lambda: int = 20,
        lambda_min_ratio: float = 0.01,
    ) -> list[RegularizedMIRTResult]:
        """Fit models for a sequence of lambda values.

        Parameters
        ----------
        responses : NDArray
            Response matrix.
        lambda_values : list, optional
            Explicit lambda values. If None, generates geometric sequence.
        n_lambda : int
            Number of lambda values if generating sequence.
        lambda_min_ratio : float
            Ratio of min to max lambda.

        Returns
        -------
        list[RegularizedMIRTResult]
            Results for each lambda value, from largest to smallest.
        """
        responses = self._validated_binary_responses(responses)
        if lambda_values is None:
            n_lambda = _validate_integer(n_lambda, "n_lambda", minimum=1)
            lambda_min_ratio = _validate_real(
                lambda_min_ratio,
                "lambda_min_ratio",
                minimum=0.0,
                inclusive=False,
            )
            if lambda_min_ratio > 1.0:
                raise ValueError("lambda_min_ratio must be at most 1")
            lambda_max = self._compute_lambda_max(responses)
            lambda_min = lambda_max * lambda_min_ratio
            if lambda_max == 0.0:
                lambda_values = [0.0] * n_lambda
            else:
                lambda_values = np.geomspace(
                    lambda_max,
                    lambda_min,
                    n_lambda,
                ).tolist()
        elif len(lambda_values) == 0:
            raise ValueError("lambda_values must contain at least one value")
        else:
            lambda_values = [
                _validate_real(
                    value,
                    "lambda_values",
                    minimum=0.0,
                    inclusive=True,
                )
                for value in lambda_values
            ]

        results = []
        for lam in lambda_values:
            if self.verbose:
                print(f"\n--- Lambda = {lam:.6f} ---")
            result = self.fit(responses, lambda_val=lam)
            results.append(result)

        return results

    def _e_step(
        self,
        responses: NDArray[np.int_],
        loadings: NDArray[np.float64],
        intercepts: NDArray[np.float64],
        latent_density: LatentDensity,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute posterior weights via quadrature."""
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights
        n_persons = responses.shape[0]
        n_items = responses.shape[1]
        n_quad = len(quad_weights)

        correct, observed = _binary_response_components(responses)
        incorrect = observed - correct
        chunk_size = max(
            1,
            min(
                n_quad,
                _MAX_ESTEP_TEMP_ENTRIES // max(1, n_persons + n_items),
            ),
        )
        log_likelihoods = np.empty((n_persons, n_quad), dtype=np.float64)

        for start in range(0, n_quad, chunk_size):
            stop = min(start + chunk_size, n_quad)
            z = quad_points[start:stop] @ loadings.T + intercepts[None, :]
            probabilities = sigmoid(z)
            probabilities = np.clip(
                probabilities,
                PROB_EPSILON,
                1 - PROB_EPSILON,
            )
            log_likelihoods[:, start:stop] = (
                correct @ np.log(probabilities).T
                + incorrect @ np.log1p(-probabilities).T
            )

        log_prior_mass = latent_density.log_quadrature_mass(quad_points, quad_weights)
        log_joint = log_likelihoods + log_prior_mass[None, :]
        log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
        log_posterior = log_joint - log_marginal

        posterior_weights = np.exp(log_posterior)
        marginal_ll = np.exp(log_marginal.ravel())

        return posterior_weights, marginal_ll

    def _m_step_penalized(
        self,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        loadings: NDArray[np.float64],
        intercepts: NDArray[np.float64],
        valid_masks: list[NDArray[np.bool_]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """M-step with penalization via coordinate descent."""
        from mirt._backend_config import should_use_rust

        quad_points = self._quadrature.nodes
        n_items = responses.shape[1]
        n_factors = self.n_factors

        r_k_all, n_k_all = _expected_item_counts(responses, posterior_weights)

        if should_use_rust():
            try:
                from mirt._rust_backend import coordinate_descent_mstep_regularized

                result = coordinate_descent_mstep_regularized(
                    r_k_all,
                    n_k_all,
                    quad_points,
                    loadings,
                    intercepts,
                    self._adaptive_weights,
                    self.penalty.lambda_val,
                    self.penalty.alpha,
                    self.cd_max_iter,
                    self.cd_tol,
                )
                if result is not None:
                    return result
            except (ImportError, AttributeError):
                pass

        new_loadings = loadings.copy()
        new_intercepts = intercepts.copy()

        for _ in range(self.cd_max_iter):
            loadings_old = new_loadings.copy()

            for j in range(n_items):
                r_k = r_k_all[j]
                n_k = n_k_all[j]

                for k in range(n_factors):
                    partial_z = np.dot(quad_points, new_loadings[j]) + new_intercepts[j]
                    partial_z -= quad_points[:, k] * new_loadings[j, k]

                    p_partial = sigmoid(partial_z)
                    p_partial = np.clip(p_partial, PROB_EPSILON, 1 - PROB_EPSILON)

                    residual = r_k - n_k * p_partial
                    x_k = quad_points[:, k]

                    gradient = np.sum(residual * x_k)
                    hessian = -np.sum(n_k * p_partial * (1 - p_partial) * x_k * x_k)

                    if abs(hessian) < PROB_EPSILON:
                        continue

                    unpenalized = new_loadings[j, k] - gradient / hessian

                    adaptive_weight = self._adaptive_weights[j, k]
                    lam_eff = self.penalty.lambda_val * adaptive_weight

                    if self.penalty.type == "lasso":
                        new_loadings[j, k] = self._soft_threshold(
                            unpenalized, lam_eff / (-hessian)
                        )
                    elif self.penalty.type == "ridge":
                        new_loadings[j, k] = unpenalized / (
                            1 + 2 * lam_eff / (-hessian)
                        )
                    else:
                        lasso_part = self.penalty.alpha * lam_eff
                        ridge_part = (1 - self.penalty.alpha) * lam_eff
                        shrunk = self._soft_threshold(
                            unpenalized, lasso_part / (-hessian)
                        )
                        new_loadings[j, k] = shrunk / (1 + 2 * ridge_part / (-hessian))

                z = np.dot(quad_points, new_loadings[j]) + new_intercepts[j]
                p = sigmoid(z)
                p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)

                residual = r_k - n_k * p
                gradient = np.sum(residual)
                hessian = -np.sum(n_k * p * (1 - p))

                if abs(hessian) > PROB_EPSILON:
                    new_intercepts[j] -= gradient / hessian

            if np.max(np.abs(new_loadings - loadings_old)) < self.cd_tol:
                break

        return new_loadings, new_intercepts

    def _soft_threshold(self, x: float, lam: float) -> float:
        """Soft-thresholding operator for LASSO."""
        if x > lam:
            return x - lam
        elif x < -lam:
            return x + lam
        else:
            return 0.0

    def _compute_penalty(self, loadings: NDArray[np.float64]) -> float:
        """Compute penalty term."""
        weighted_loadings = loadings * self._adaptive_weights
        lam = self.penalty.lambda_val

        if self.penalty.type == "lasso":
            return lam * np.sum(np.abs(weighted_loadings))
        elif self.penalty.type == "ridge":
            return lam * np.sum(weighted_loadings**2)
        else:
            lasso_part = self.penalty.alpha * np.sum(np.abs(weighted_loadings))
            ridge_part = (1 - self.penalty.alpha) * np.sum(weighted_loadings**2)
            return lam * (lasso_part + ridge_part)

    def _compute_lambda_max(self, responses: NDArray[np.int_]) -> float:
        """Compute maximum lambda that gives all-zero solution."""
        from mirt.estimation.latent_density import GaussianDensity
        from mirt.models.multidimensional import MultidimensionalModel

        n_items = responses.shape[1]

        model = MultidimensionalModel(n_items=n_items, n_factors=self.n_factors)
        model._initialize_parameters()

        self._quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=self.n_factors,
        )

        latent_density = GaussianDensity(
            mean=np.zeros(self.n_factors),
            cov=np.eye(self.n_factors),
            n_dimensions=self.n_factors,
        )

        loadings = np.zeros((n_items, self.n_factors))
        intercepts = np.zeros(n_items)

        posterior_weights, _ = self._e_step(
            responses, loadings, intercepts, latent_density
        )

        quad_points = self._quadrature.nodes

        r_k_all, n_k_all = _expected_item_counts(responses, posterior_weights)
        probabilities = np.clip(
            sigmoid(intercepts),
            PROB_EPSILON,
            1 - PROB_EPSILON,
        )
        residuals = r_k_all - n_k_all * probabilities[:, None]
        max_gradient = float(np.max(np.abs(residuals @ quad_points), initial=0.0))
        return max_gradient * 1.1

    def _count_possible_models(self, n_items: int, n_factors: int) -> int:
        """Count possible loading patterns for EBIC."""
        return 2 ** (n_items * n_factors)

    def _fit_unpenalized_warmstart(
        self,
        responses: NDArray[np.int_],
        loadings: NDArray[np.float64],
        intercepts: NDArray[np.float64],
        latent_density: LatentDensity,
        max_iter: int = 50,
    ) -> NDArray[np.float64]:
        """Fit unpenalized model for adaptive weights."""
        valid_masks = [responses[:, j] >= 0 for j in range(responses.shape[1])]

        for _ in range(max_iter):
            posterior_weights, _ = self._e_step(
                responses, loadings, intercepts, latent_density
            )

            original_lambda = self.penalty.lambda_val
            self.penalty.lambda_val = 0.0
            self._adaptive_weights = np.ones_like(loadings)

            loadings, intercepts = self._m_step_penalized(
                responses, posterior_weights, loadings, intercepts, valid_masks
            )

            self.penalty.lambda_val = original_lambda

            n_k = posterior_weights.sum(axis=0)
            latent_density.update(self._quadrature.nodes, n_k)

        return loadings
