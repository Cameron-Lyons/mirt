from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from math import lgamma
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt._core import sigmoid
from mirt._logistic import _affine_probability
from mirt.constants import PROB_EPSILON, REGULARIZATION_EPSILON
from mirt.estimation._em_context import EMFitContext
from mirt.estimation._posterior import normalize_log_posterior
from mirt.estimation.base import BaseEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.exceptions import MirtDataError, MirtValidationError

if TYPE_CHECKING:
    from mirt.estimation.latent_density import LatentDensity
    from mirt.models.multidimensional import MultidimensionalModel

_PENALTY_TYPES = frozenset(("lasso", "ridge", "elastic_net"))
_MAX_ESTEP_TEMP_ENTRIES = 1_000_000
_EBIC_GAMMA = 0.5
# EM iterations of the unpenalized reference fit. It only provides starting
# values, adaptive weights and the path scale: after the first few dozen
# iterations EM merely drifts along the rotation, on which the likelihood is
# flat up to quadrature error.
_REFERENCE_MAX_ITER = 100


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


def ebic_penalty(
    n_candidates: int,
    n_selected: int,
    gamma: float = _EBIC_GAMMA,
) -> float:
    """Return the model-space term that the extended BIC adds to BIC.

    Chen and Chen (2008) weight every model with ``k`` of ``P`` candidate
    parameters by the number ``C(P, k)`` of such models, so that
    ``EBIC = BIC + 2 * gamma * log C(P, k)``. Unlike a constant, the term
    depends on the model size and changes which sparse model is selected.

    Parameters
    ----------
    n_candidates : int
        Number ``P`` of candidate parameters, such as every loading.
    n_selected : int
        Number ``k`` of selected (non-zero) candidates, ``0 <= k <= P``.
    gamma : float, default=0.5
        Weight of the model-space term in ``[0, 1]``; 0 gives BIC.

    Returns
    -------
    float
        ``2 * gamma * log C(P, k)``, and 0 without candidates.

    References
    ----------
    Chen, J., & Chen, Z. (2008). Extended Bayesian information criteria for
        model selection with large model spaces. Biometrika, 95, 759-771.
    """
    n_candidates = int(n_candidates)
    n_selected = int(n_selected)
    if n_candidates <= 0:
        return 0.0
    if not 0 <= n_selected <= n_candidates:
        raise MirtValidationError(
            "n_selected must lie between 0 and n_candidates",
            parameter="n_selected",
            value=n_selected,
            expected=f"[0, {n_candidates}]",
        )
    log_count = (
        lgamma(n_candidates + 1)
        - lgamma(n_selected + 1)
        - lgamma(n_candidates - n_selected + 1)
    )
    return 2.0 * float(gamma) * log_count


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

    @property
    def l1_share(self) -> float:
        """Share of ``lambda_val`` on the absolute loadings.

        ``"lasso"`` is a pure L1 penalty and ``"ridge"`` a pure L2 penalty
        whatever ``alpha`` holds; ``alpha`` mixes the two for
        ``"elastic_net"`` only.
        """
        if self.type == "lasso":
            return 1.0
        if self.type == "ridge":
            return 0.0
        return self.alpha


@dataclass
class RegularizedMIRTResult:
    """Result from regularized MIRT estimation.

    ``log_likelihood`` is the marginal log-likelihood at the estimates and
    ``penalized_ll`` subtracts the penalty from it. AIC, BIC and EBIC count
    the non-zero loadings and the intercepts. EBIC adds
    ``2 * 0.5 * log C(n_items * n_factors, n_nonzero)`` to BIC (Chen and
    Chen, 2008; see :func:`ebic_penalty`).
    """

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


RegularizedStart: TypeAlias = (
    Literal["default"] | RegularizedMIRTResult | Mapping[str, ArrayLike]
)


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
        Elastic net mixing parameter (1 = pure LASSO, 0 = pure ridge). Used
        by ``penalty="elastic_net"`` only.
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

    Notes
    -----
    The penalized likelihood is not concave. With every loading at zero the
    posterior equals the prior, so zero loadings are a stationary point for
    every penalty, and equal starting loadings on every factor are kept equal
    by the coordinate updates, which traps a fit in one factor duplicated.
    Each fit therefore starts from a reference solution: an unpenalized EM
    fit from loadings staggered across factors (item ``j`` loads 0.8 on
    factor ``j mod n_factors`` and 0.4 on the others), rotated to varimax
    simple structure. The rotation is orthogonal and leaves the likelihood
    unchanged because the factors are independent standard normals. The
    reference solution also gives the adaptive-LASSO weights
    ``1 / |a_jk|`` and the largest penalty of the default path (see
    :meth:`fit_path`).
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
        self._fit_context: EMFitContext | None = None
        self._reference_cache: (
            tuple[tuple[object, ...], NDArray[np.float64], NDArray[np.float64]] | None
        ) = None

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

    @contextmanager
    def _prepared(self, responses: NDArray[np.int_]) -> Iterator[None]:
        """Share one response preparation and quadrature grid within a call."""
        previous_context = self._fit_context
        if previous_context is not None and previous_context.responses is responses:
            yield
            return
        with EMFitContext(responses) as context:
            self._fit_context = context
            self._quadrature = GaussHermiteQuadrature(
                n_points=self.n_quadpts,
                n_dimensions=self.n_factors,
            )
            try:
                yield
            finally:
                self._fit_context = previous_context

    def fit(
        self,
        responses: NDArray[np.int_],
        lambda_val: float | None = None,
        *,
        start: RegularizedStart = "default",
    ) -> RegularizedMIRTResult:
        """Fit regularized MIRT model.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items). Missing values coded as -1.
        lambda_val : float, optional
            Override regularization strength.
        start : "default", RegularizedMIRTResult or mapping, default="default"
            Starting loadings and intercepts. ``"default"`` starts from the
            varimax-rotated unpenalized solution (see the class notes). A
            previous result, such as the fit at a neighbouring penalty, or a
            mapping with ``"loadings"`` and ``"intercepts"`` arrays
            warm-starts the fit. Adaptive weights always come from the
            unpenalized solution.

        Returns
        -------
        RegularizedMIRTResult
            Fitted model with sparse loadings.

        Raises
        ------
        MirtValidationError
            If ``start`` is not ``"default"``, a result or a mapping of
            finite arrays with the shapes of the loadings and intercepts.
        """
        responses = self._validated_binary_responses(responses)
        initial = self._resolve_start(start, responses.shape[1])
        with self._prepared(responses):
            return self._fit_prepared(responses, lambda_val, initial)

    def _resolve_start(
        self,
        start: RegularizedStart,
        n_items: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Return starting arrays, or ``None`` for the reference solution."""
        if isinstance(start, str) and start == "default":
            return None
        if isinstance(start, RegularizedMIRTResult):
            values: Mapping[str, ArrayLike] = {
                "loadings": start.loadings,
                "intercepts": start.intercepts,
            }
        elif isinstance(start, Mapping) and set(start) == {"loadings", "intercepts"}:
            values = start
        else:
            raise MirtValidationError(
                "start must be 'default', a RegularizedMIRTResult, or a mapping "
                "with 'loadings' and 'intercepts'",
                parameter="start",
                value=type(start).__name__,
            )
        shapes = {"loadings": (n_items, self.n_factors), "intercepts": (n_items,)}
        arrays = []
        for name, shape in shapes.items():
            try:
                array = np.array(values[name], dtype=np.float64, copy=True)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    f"starting {name} must be numeric", parameter="start"
                ) from exc
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise MirtValidationError(
                    f"starting {name} must be finite with shape {shape}",
                    parameter="start",
                    value=array.shape,
                    expected=str(shape),
                )
            arrays.append(array)
        return arrays[0], arrays[1]

    def _fit_prepared(
        self,
        responses: NDArray[np.int_],
        lambda_val: float | None,
        start: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    ) -> RegularizedMIRTResult:
        """Fit using response preparation owned by the current call."""
        from mirt.estimation.latent_density import GaussianDensity
        from mirt.models.multidimensional import MultidimensionalModel

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

        latent_density = GaussianDensity(
            mean=np.zeros(self.n_factors),
            cov=np.eye(self.n_factors),
            n_dimensions=self.n_factors,
        )

        reference_loadings, reference_intercepts = self._reference_solution(responses)
        if start is None:
            loadings, intercepts = reference_loadings, reference_intercepts
        else:
            loadings, intercepts = (array.copy() for array in start)

        if self.adaptive:
            self._adaptive_weights = 1.0 / (
                np.abs(reference_loadings) + REGULARIZATION_EPSILON
            )
        else:
            self._adaptive_weights = np.ones_like(loadings)

        self._convergence_history = []
        prev_ll = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            posterior_weights, log_marginal = self._e_step(
                responses, loadings, intercepts, latent_density
            )

            current_ll = float(log_marginal.sum())
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
                responses, posterior_weights, loadings, intercepts
            )

            n_k = posterior_weights.sum(axis=0)
            del posterior_weights
            latent_density.update(self._quadrature.nodes, n_k)
        else:
            posterior_weights, log_marginal = self._e_step(
                responses, loadings, intercepts, latent_density
            )
            current_ll = float(log_marginal.sum())
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
        ebic = bic + ebic_penalty(loadings.size, n_nonzero)

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

        The penalties are fitted from the smallest to the largest, each
        starting from the fit at the next smaller penalty, and the smallest
        from the reference solution (see the class notes). Increasing the
        penalty therefore removes loadings from a solution that already has
        simple structure. The reverse order cannot work: once every loading
        of a factor is zero, that factor's posterior equals its prior and its
        loadings stay zero at every smaller penalty.

        Parameters
        ----------
        responses : NDArray
            Response matrix.
        lambda_values : list, optional
            Explicit lambda values. If None, generates a geometric sequence
            of ``n_lambda`` values from ``lambda_max`` down to
            ``lambda_max * lambda_min_ratio``.
        n_lambda : int
            Number of lambda values if generating sequence.
        lambda_min_ratio : float
            Ratio of min to max lambda.

        Returns
        -------
        list[RegularizedMIRTResult]
            Results in the order of ``lambda_values``; a generated sequence
            runs from the largest penalty to the smallest.

        Notes
        -----
        ``lambda_max`` is the smallest L1 penalty at which the first M-step
        from the reference solution sets every loading to zero. That M-step
        maximizes, for each item, a concave expected complete-data
        log-likelihood over the reference posterior, so by its KKT conditions
        ``lambda_max = max_jk |sum_i (y_ij - p_j) E[theta_ik | y_i]| /
        (w_jk * l1_share)``, where ``p_j`` is the item's observed proportion
        correct, ``w_jk`` the adaptive weight (1 without ``adaptive``) and
        ``l1_share`` is ``alpha`` for the elastic net and 1 otherwise. At and
        above ``lambda_max`` the fit is all zero; just below it the first
        M-step keeps the loading attaining the maximum. Because zero loadings
        are a stationary point, a converged fit can still collapse to zero
        somewhat below ``lambda_max``, so the largest penalties of a
        generated path may fit the null model. A ridge penalty never zeroes
        loadings, and its path uses the LASSO value as its scale.
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

        original_lambda = self.penalty.lambda_val
        with self._prepared(responses):
            if lambda_values is None:
                lambda_max = self._compute_lambda_max(responses)
                if lambda_max == 0.0:
                    lambda_values = [0.0] * n_lambda
                else:
                    lambda_values = np.geomspace(
                        lambda_max,
                        lambda_max * lambda_min_ratio,
                        n_lambda,
                    ).tolist()

            results: list[RegularizedMIRTResult | None] = [None] * len(lambda_values)
            start = None
            try:
                for index in np.argsort(lambda_values, kind="stable"):
                    lam = lambda_values[index]
                    if self.verbose:
                        print(f"\n--- Lambda = {lam:.6f} ---")
                    result = self._fit_prepared(responses, lam, start)
                    results[index] = result
                    start = (result.loadings, result.intercepts)
            finally:
                self.penalty.lambda_val = original_lambda

        return [result for result in results if result is not None]

    def _e_step(
        self,
        responses: NDArray[np.int_],
        loadings: NDArray[np.float64],
        intercepts: NDArray[np.float64],
        latent_density: LatentDensity,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return posterior weights and log marginals with bounded scratch space."""
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights
        n_persons = responses.shape[0]
        n_items = responses.shape[1]
        n_quad = len(quad_weights)

        context = self._response_context(responses)
        chunk_size = max(
            1,
            min(
                n_quad,
                _MAX_ESTEP_TEMP_ENTRIES // max(1, 2 * n_persons + 8 * n_items),
            ),
        )
        log_joint = np.empty((n_persons, n_quad), dtype=np.float64)

        for start in range(0, n_quad, chunk_size):
            stop = min(start + chunk_size, n_quad)
            probabilities = _affine_probability(
                quad_points[start:stop], loadings, intercepts
            )
            np.clip(probabilities, PROB_EPSILON, 1 - PROB_EPSILON, out=probabilities)
            log_correct = np.log(probabilities)
            np.negative(probabilities, out=probabilities)
            np.log1p(probabilities, out=probabilities)
            row_block = max(
                1, _MAX_ESTEP_TEMP_ENTRIES // max(1, 3 * n_items + 2 * (stop - start))
            )
            for row in range(0, n_persons, row_block):
                row_stop = min(row + row_block, n_persons)
                correct, observed = context.response_components(row, row_stop)
                target = log_joint[row:row_stop, start:stop]
                np.matmul(correct, log_correct.T, out=target)
                target += (observed - correct) @ probabilities.T

        log_prior_mass = latent_density.log_quadrature_mass(quad_points, quad_weights)
        return normalize_log_posterior(log_joint, log_prior_mass)

    def _response_context(self, responses: NDArray[np.int_]) -> EMFitContext:
        """Reuse current-fit preparation and isolate independent private calls."""
        if self._fit_context is not None and self._fit_context.responses is responses:
            return self._fit_context
        return EMFitContext(responses)

    def _m_step_penalized(
        self,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        loadings: NDArray[np.float64],
        intercepts: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """M-step with penalization via coordinate descent.

        Each coordinate takes a Newton step on the expected complete-data
        log-likelihood at its current value, followed by the proximal
        operator of its penalty.
        """
        from mirt._backend_config import should_use_rust

        quad_points = self._quadrature.nodes
        n_items = responses.shape[1]
        n_factors = self.n_factors

        r_k_all, n_k_all = self._response_context(responses).expected_counts(
            posterior_weights
        )
        # The native kernel selects its penalty by the L1 share alone.
        l1_share = self.penalty.l1_share

        if should_use_rust():
            try:
                from mirt.backends.rust.regularized import (
                    coordinate_descent_mstep_regularized,
                )

                result = coordinate_descent_mstep_regularized(
                    r_k_all,
                    n_k_all,
                    quad_points,
                    loadings,
                    intercepts,
                    self._adaptive_weights,
                    self.penalty.lambda_val,
                    l1_share,
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
                    z = np.dot(quad_points, new_loadings[j]) + new_intercepts[j]
                    p = sigmoid(z)
                    p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)

                    residual = r_k - n_k * p
                    x_k = quad_points[:, k]

                    gradient = np.sum(residual * x_k)
                    hessian = -np.sum(n_k * p * (1 - p) * x_k * x_k)

                    if abs(hessian) < PROB_EPSILON:
                        continue

                    unpenalized = new_loadings[j, k] - gradient / hessian

                    adaptive_weight = self._adaptive_weights[j, k]
                    lam_eff = self.penalty.lambda_val * adaptive_weight
                    lasso_part = l1_share * lam_eff
                    ridge_part = (1 - l1_share) * lam_eff
                    shrunk = self._soft_threshold(unpenalized, lasso_part / (-hessian))
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
        """Compute the penalty that the M-step's proximal updates apply.

        ``lambda * sum(w * (l1_share * |a| + (1 - l1_share) * a**2))`` with
        the adaptive weights ``w``.
        """
        l1_share = self.penalty.l1_share
        absolute = np.abs(loadings)
        weighted = self._adaptive_weights * (
            l1_share * absolute + (1 - l1_share) * absolute**2
        )
        return float(self.penalty.lambda_val * np.sum(weighted))

    def _compute_lambda_max(self, responses: NDArray[np.int_]) -> float:
        """Return the smallest penalty whose first M-step zeroes every loading.

        See the notes of :meth:`fit_path`. The gradient is that of each
        item's expected complete-data log-likelihood under the posterior of
        the reference solution, at zero loadings and the intercept that
        reproduces the item's observed proportion correct.
        """
        from mirt.estimation.latent_density import GaussianDensity

        with self._prepared(responses):
            loadings, intercepts = self._reference_solution(responses)
            posterior_weights, _ = self._e_step(
                responses,
                loadings,
                intercepts,
                GaussianDensity(
                    mean=np.zeros(self.n_factors),
                    cov=np.eye(self.n_factors),
                    n_dimensions=self.n_factors,
                ),
            )
            r_k_all, n_k_all = self._response_context(responses).expected_counts(
                posterior_weights
            )
            quad_points = self._quadrature.nodes
        n_observed = n_k_all.sum(axis=1)
        proportions = np.divide(
            r_k_all.sum(axis=1),
            n_observed,
            out=np.zeros(len(n_observed)),
            where=n_observed > 0.0,
        )
        gradient = np.abs((r_k_all - n_k_all * proportions[:, None]) @ quad_points)
        if self.adaptive:
            gradient *= np.abs(loadings) + REGULARIZATION_EPSILON
        l1_share = self.penalty.l1_share
        scale = l1_share if l1_share > 0.0 else 1.0
        return float(np.max(gradient, initial=0.0) / scale)

    def _reference_key(self, responses: NDArray[np.int_]) -> tuple[object, ...]:
        """Identify the data and settings that determine the reference fit."""
        data = np.ascontiguousarray(responses)
        digest = hashlib.blake2b(data.view(np.uint8), digest_size=16).hexdigest()
        return (
            data.shape,
            data.dtype.str,
            digest,
            self.n_factors,
            self.n_quadpts,
            self.max_iter,
            self.tol,
            self.cd_max_iter,
            self.cd_tol,
        )

    def _reference_solution(
        self,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the varimax-rotated unpenalized fit, computed once per data set.

        EM starts from loadings staggered across factors and zero intercepts,
        as exploratory EM does, and runs for at most ``min(max_iter, 100)``
        iterations. Columns are
        signed to have a non-negative sum and ordered by their sum of
        squares.
        """
        from mirt.estimation.base import _stagger_factor_slopes
        from mirt.estimation.latent_density import GaussianDensity
        from mirt.models.multidimensional import MultidimensionalModel
        from mirt.utils.rotation import rotate_loadings

        key = self._reference_key(responses)
        if self._reference_cache is not None and self._reference_cache[0] == key:
            return self._reference_cache[1].copy(), self._reference_cache[2].copy()

        n_items = responses.shape[1]
        model = MultidimensionalModel(n_items=n_items, n_factors=self.n_factors)
        model._initialize_parameters()
        loadings = _stagger_factor_slopes(
            model, "slopes", model.slopes, model.free_parameter_masks["slopes"]
        ).copy()
        intercepts = model.intercepts.copy()
        latent_density = GaussianDensity(
            mean=np.zeros(self.n_factors),
            cov=np.eye(self.n_factors),
            n_dimensions=self.n_factors,
        )

        original_lambda = self.penalty.lambda_val
        original_weights = self._adaptive_weights
        self.penalty.lambda_val = 0.0
        self._adaptive_weights = np.ones_like(loadings)
        try:
            prev_ll = -np.inf
            for _ in range(min(self.max_iter, _REFERENCE_MAX_ITER)):
                posterior_weights, log_marginal = self._e_step(
                    responses, loadings, intercepts, latent_density
                )
                current_ll = float(log_marginal.sum())
                if self._check_convergence(prev_ll, current_ll):
                    break
                prev_ll = current_ll
                loadings, intercepts = self._m_step_penalized(
                    responses, posterior_weights, loadings, intercepts
                )
                n_k = posterior_weights.sum(axis=0)
                del posterior_weights
                latent_density.update(self._quadrature.nodes, n_k)
        finally:
            self.penalty.lambda_val = original_lambda
            self._adaptive_weights = original_weights

        if n_items >= self.n_factors and np.all(np.isfinite(loadings)):
            loadings = rotate_loadings(loadings, method="varimax", normalize=True)[0]
        loadings = loadings * np.where(loadings.sum(axis=0) < 0.0, -1.0, 1.0)
        loadings = loadings[:, np.argsort(-np.sum(loadings**2, axis=0), kind="stable")]
        self._reference_cache = (key, loadings.copy(), intercepts.copy())
        return loadings, intercepts
