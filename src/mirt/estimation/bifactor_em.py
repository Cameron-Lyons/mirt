"""Bifactor EM estimation by Gibbons-Hedeker dimension reduction.

A bifactor item depends on the general factor and on one specific factor.
With independent standard-normal factors, the specific factors are
conditionally independent given the general one, so the marginal likelihood
of a response pattern factors as

.. math::

   L = \\sum_g w_g \\prod_s \\sum_k w_k \\prod_{j \\in s} P(y_j \\mid g, k).

Each specific factor therefore needs only a two-dimensional (general by
specific) quadrature grid instead of the ``(1 + S)``-dimensional product grid,
and the result equals the product-grid quadrature exactly. The cost of an
E-step falls from ``O(N J Q^(1 + S))`` to ``O(N J Q^2)``. Graded items enter
through their category probabilities, so the same reduction fits graded
bifactor models (Gibbons et al., 2007).

References
----------
Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
    analysis. Psychometrika, 57(3), 423-436.
Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
    D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
    Full-information item bifactor analysis of graded response data. Applied
    Psychological Measurement, 31(1), 4-19.
Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information item
    bifactor analysis. Psychological Methods, 16(3), 221-248.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit

from mirt._model_defaults import uses_builtin_model_hooks
from mirt.constants import PROB_EPSILON
from mirt.estimation._em_context import EMFitContext
from mirt.estimation._graded_order import THRESHOLD_GAP, GradedOrder, graded_order
from mirt.estimation._logistic_newton import newton_logistic_items
from mirt.estimation.base import (
    BaseEstimator,
    StartValues,
    _apply_starting_values,
    _parameter_bounds,
    _validate_start,
)
from mirt.estimation.quadrature import (
    QuadratureRule,
    build_quadrature,
    validate_quadrature_rule,
)
from mirt.estimation.standard_errors import StandardErrorMethod, validate_se_method
from mirt.exceptions import MirtDataError, MirtModelError, MirtValidationError

if TYPE_CHECKING:
    from mirt.estimation.em import EMEstimator
    from mirt.models.bifactor import BifactorModel, GradedBifactorModel
    from mirt.models.testlet import TestletModel
    from mirt.results.fit_result import FitResult

# Stored parameters of a bifactor item, in model and design-column order.
_PARAMETERS = ("general_loadings", "specific_loadings", "intercepts")
# Person-by-node scratch entries for one block of response rows.
_MAX_BLOCK_ENTRIES = 1 << 22
# Bounded fallback M-step for items with fixed coordinates or Newton failures.
_ITEM_MAXITER = 200
_ITEM_FTOL = 1e-12


@dataclass(frozen=True)
class _Grid:
    """One-dimensional standard-normal rule and its (general, specific) grid.

    ``design`` rows are ``(general, specific, 1)`` at node ``g * Q + k``, so
    an item's logits on the grid are ``design @ (a_g, a_s, d)``. The rule is
    Gauss-Hermite or, as in R mirt, ``Q`` equally spaced nodes on ``[-6, 6]``
    with normal-density weights; both factor into the product grid.
    """

    nodes: NDArray[np.float64]
    log_weights: NDArray[np.float64]
    design: NDArray[np.float64]
    groups: tuple[NDArray[np.intp], ...]

    @classmethod
    def build(
        cls,
        n_points: int,
        model: BifactorModel | GradedBifactorModel,
        rule: QuadratureRule = "gauss-hermite",
    ) -> _Grid:
        indices = model._specific_factor_indices
        groups = tuple(
            np.flatnonzero(indices == factor)
            for factor in range(model.n_specific_factors)
        )
        return cls.from_groups(n_points, groups, rule)

    @classmethod
    def from_groups(
        cls,
        n_points: int,
        groups: tuple[NDArray[np.intp], ...],
        rule: QuadratureRule = "gauss-hermite",
    ) -> _Grid:
        """Build the grid for items grouped by specific factor."""
        quadrature = build_quadrature(n_points, 1, rule)
        nodes = quadrature.nodes.ravel()
        weights = quadrature.weights
        general, specific = np.meshgrid(nodes, nodes, indexing="ij")
        design = np.column_stack(
            (general.ravel(), specific.ravel(), np.ones(n_points * n_points))
        )
        return cls(nodes, np.log(weights / weights.sum()), design, groups)

    @property
    def n_points(self) -> int:
        return self.nodes.size

    def block_size(self, width: int) -> int:
        """Rows per block when each row holds ``width`` grid-sized arrays."""
        return max(1, _MAX_BLOCK_ENTRIES // (self.design.shape[0] * max(1, width)))


@dataclass(frozen=True)
class _Curves:
    """Clipped item probabilities on the grid, as the model's likelihood uses."""

    probability: NDArray[np.float64]
    log_correct: NDArray[np.float64]
    log_incorrect: NDArray[np.float64]
    active: NDArray[np.bool_]

    @classmethod
    def evaluate(cls, grid: _Grid, coefficients: NDArray[np.float64]) -> _Curves:
        raw = expit(grid.design @ coefficients.T)
        active = (raw > PROB_EPSILON) & (raw < 1.0 - PROB_EPSILON)
        clipped = np.clip(raw, PROB_EPSILON, 1.0 - PROB_EPSILON)
        return cls(raw, np.log(clipped), np.log1p(-clipped), active)


@dataclass
class _Block:
    """Posterior of one block of response rows on the reduced grids.

    ``general`` has shape ``(n, Q)`` and each entry of ``conditionals`` holds
    a specific factor's posterior given the general node, ``(n, Q, Q)``.
    """

    start: int
    stop: int
    correct: NDArray[np.float64]
    observed: NDArray[np.float64]
    log_marginal: NDArray[np.float64]
    general: NDArray[np.float64]
    conditionals: list[NDArray[np.float64]]

    def joint(self, factor: int) -> NDArray[np.float64]:
        """Return the ``(n, Q * Q)`` posterior of (general, specific) nodes."""
        joint = self.general[:, :, None] * self.conditionals[factor]
        return joint.reshape(joint.shape[0], -1)


def _coefficients(model: BifactorModel) -> NDArray[np.float64]:
    """Return item coefficients ``(a_g, a_s, d)`` as rows."""
    return np.column_stack([model._parameters[name] for name in _PARAMETERS])


def _reduced_posterior(
    grid: _Grid,
    n_rows: int,
    log_joints: Iterable[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]:
    """Normalize per-factor joint log-likelihoods into reduced posteriors.

    Each ``(n, Q, Q)`` entry of ``log_joints`` holds ``log P(y_s | g, k) +
    log w_k`` for one specific factor and is overwritten. Returns the
    marginal log-likelihoods ``(n,)``, the general-factor posterior
    ``(n, Q)`` and each factor's conditional posterior given the general
    node.
    """
    log_general = np.tile(grid.log_weights, (n_rows, 1))
    conditionals = []
    for log_joint in log_joints:
        shift = log_joint.max(axis=2, keepdims=True)
        log_joint -= shift
        conditional = np.exp(log_joint, out=log_joint)
        total = conditional.sum(axis=2, keepdims=True)
        conditional /= total
        log_general += (shift + np.log(total))[:, :, 0]
        conditionals.append(conditional)
    shift = log_general.max(axis=1, keepdims=True)
    general = np.exp(log_general - shift)
    total = general.sum(axis=1, keepdims=True)
    general /= total
    return (shift + np.log(total))[:, 0], general, conditionals


def _posterior_blocks(
    grid: _Grid,
    curves: _Curves,
    context: EMFitContext,
    block_size: int,
) -> Iterator[_Block]:
    """Yield exact reduced posteriors for consecutive blocks of response rows."""
    n_points = grid.n_points
    n_rows = context.responses.shape[0]
    # One product per specific factor gives log P(y_s | g, k) + log w_k from
    # the stacked (correct, observed, 1) response columns.
    difference = curves.log_correct - curves.log_incorrect
    log_weights = np.tile(grid.log_weights, n_points)
    tables = [
        np.vstack(
            (difference[:, items].T, curves.log_incorrect[:, items].T, log_weights)
        )
        for items in grid.groups
    ]
    for start in range(0, n_rows, block_size):
        stop = min(start + block_size, n_rows)
        correct, observed = context.response_components(start, stop)
        n = stop - start
        ones = np.ones((n, 1))
        log_marginal, general, conditionals = _reduced_posterior(
            grid,
            n,
            (
                (
                    np.hstack((correct[:, items], observed[:, items], ones)) @ table
                ).reshape(n, n_points, n_points)
                for items, table in zip(grid.groups, tables, strict=True)
            ),
        )
        yield _Block(
            start, stop, correct, observed, log_marginal, general, conditionals
        )


def _row_weights(
    frequencies: NDArray[np.float64] | None, start: int, stop: int
) -> NDArray[np.float64]:
    if frequencies is None:
        return np.ones(stop - start)
    return frequencies[start:stop]


def _expected_counts(
    grid: _Grid,
    curves: _Curves,
    context: EMFitContext,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Run an E-step and return the log-likelihood and expected item counts.

    Item ``j`` of specific factor ``s`` receives counts on the ``(Q * Q)``
    grid of the general factor and factor ``s``.
    """
    n_items = curves.probability.shape[1]
    correct_counts = np.zeros((n_items, grid.design.shape[0]))
    observed_counts = np.zeros_like(correct_counts)
    frequencies = context.frequencies
    log_likelihood = 0.0
    width = len(grid.groups) + 1
    for block in _posterior_blocks(grid, curves, context, grid.block_size(width)):
        weights = _row_weights(frequencies, block.start, block.stop)
        log_likelihood += float(weights @ block.log_marginal)
        for factor, items in enumerate(grid.groups):
            columns = np.hstack((block.correct[:, items], block.observed[:, items]))
            counts = (columns * weights[:, None]).T @ block.joint(factor)
            correct_counts[items] += counts[: items.size]
            observed_counts[items] += counts[items.size :]
    return log_likelihood, correct_counts, observed_counts


def _bounded_item_update(
    design: NDArray[np.float64],
    correct: NDArray[np.float64],
    observed: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    free: NDArray[np.bool_],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Maximize one item's expected log-likelihood over its free coordinates."""
    incorrect = observed - correct

    def objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        trial = coefficients.copy()
        trial[free] = params
        logits = design @ trial
        loss = float(
            np.sum(
                correct * np.logaddexp(0.0, -logits)
                + incorrect * np.logaddexp(0.0, logits)
            )
        )
        residual = observed * expit(logits) - correct
        return loss, (design.T @ residual)[free]

    result = minimize(
        objective,
        np.clip(coefficients[free], lower[free], upper[free]),
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(lower[free], upper[free], strict=True)),
        options={"maxiter": _ITEM_MAXITER, "ftol": _ITEM_FTOL},
    )
    updated = coefficients.copy()
    updated[free] = result.x
    return updated


class BifactorEMEstimator(BaseEstimator):
    """Exact bifactor EM on two-dimensional (general by specific) grids.

    The estimator fits a dichotomous :class:`~mirt.models.BifactorModel` or a
    graded :class:`~mirt.models.GradedBifactorModel` with independent
    standard-normal factors by marginal maximum likelihood. The
    Gibbons-Hedeker dimension reduction integrates each specific factor
    jointly with the general factor only, which reproduces the
    ``(1 + S)``-dimensional product-grid likelihood of :class:`EMEstimator`
    exactly at a cost that grows linearly rather than exponentially in the
    number of specific factors ``S``. Default settings are therefore usable
    for any number of specific factors.

    Testlet models (:class:`~mirt.models.TestletModel`,
    :class:`~mirt.models.BifactorTestletModel` and
    :class:`~mirt.models.RandomTestletEffectsModel`) are bifactor models
    whose specific slopes are ``d_j * sigma_t``, with one specific factor per
    testlet and none for standalone items, and are fitted the same way.
    Every M-step first updates each item's free coordinates with the
    testlet-level ones held, then maximizes the summed expected
    log-likelihood of a testlet's items over its common loading (with
    ``constrain_testlet_loadings=True``) or, when its loadings are fixed,
    over its standard deviation ``sigma_t`` (a conditional maximization, so
    EM stays monotone). :class:`EMEstimator` delegates testlet models here
    when its other settings allow.

    Parameters
    ----------
    n_quadpts : int, default=21
        Quadrature points per factor. Each specific factor uses a grid of
        ``n_quadpts ** 2`` (general, specific) nodes.
    max_iter : int, default=500
        Maximum number of M-steps.
    tol : float, default=1e-4
        Convergence tolerance for the change in marginal log-likelihood
        between consecutive EM iterates.
    verbose : bool, default=False
        Print the log-likelihood at each iterate.
    compute_standard_errors : bool, default=True
        Whether to compute item parameter standard errors after fitting.
    se_method : {"auto", "oakes", "crossprod", "sandwich", "complete_data"}, \
default="auto"
        Standard-error estimator, as for :class:`EMEstimator`. ``"auto"`` and
        ``"oakes"`` invert the observed information of the marginal
        likelihood, computed exactly from the reduced posteriors by Louis's
        (1982) identity, and store the parameter covariance in
        ``FitResult.vcov``. ``"crossprod"`` inverts the outer product of the
        marginal person scores and ``"sandwich"`` combines both.
        ``"complete_data"`` uses the diagonal complete-data curvature, which
        understates uncertainty. Coordinates at an optimizer bound are held
        fixed with ``NaN`` standard errors.
    quadrature : {"gauss-hermite", "rectangular", "auto"}, default="gauss-hermite"
        One-dimensional rule of the grids. ``"rectangular"`` is R mirt's
        ``n_quadpts`` equally spaced nodes on ``[-6, 6]`` with normal-density
        weights (R's ``bfactor`` uses 31); ``"auto"`` chooses it from 15
        points on (see :func:`~mirt.estimation.quadrature.build_quadrature`).

    Notes
    -----
    Every M-step solves all items without fixed coordinates jointly by
    Newton's method on the expected counts. Items with fixed coordinates, or
    whose Newton estimates leave the parameter box, use a bounded
    quasi-Newton solve. An item whose specific loading is fixed at zero loads on the
    general factor only. Graded items are solved by Newton's method on their
    expected category counts, all items with the same number of categories
    at once, with the intercepts of each item kept decreasing; fixed
    coordinates are held and items whose estimates reach the parameter box
    use a bounded SLSQP solve. Responses must be dichotomous for a
    ``BifactorModel`` or testlet model and category codes ``0, ...,
    n_categories[j] - 1`` for a ``GradedBifactorModel``, with negative codes
    for missing responses. Item priors, latent density estimation and
    correlated factors are not supported; use :class:`EMEstimator` for them.

    References
    ----------
    Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
    analysis. *Psychometrika*, 57(3), 423-436.

    Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
    Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover,
    A. (2007). Full-information item bifactor analysis of graded response
    data. *Applied Psychological Measurement*, 31(1), 4-19.

    Louis, T. A. (1982). Finding the observed information matrix when using
    the EM algorithm. *Journal of the Royal Statistical Society B*, 44(2),
    226-233.
    """

    def __init__(
        self,
        n_quadpts: int = 21,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        compute_standard_errors: bool = True,
        se_method: StandardErrorMethod = "auto",
        quadrature: QuadratureRule = "gauss-hermite",
    ) -> None:
        super().__init__(max_iter, tol, verbose)
        if (
            isinstance(n_quadpts, (bool, np.bool_))
            or not isinstance(n_quadpts, (int, np.integer))
            or n_quadpts < 5
        ):
            raise MirtValidationError(
                "n_quadpts should be an integer of at least 5",
                parameter="n_quadpts",
                value=n_quadpts,
                expected=">= 5",
            )
        if not isinstance(compute_standard_errors, (bool, np.bool_)):
            raise MirtValidationError(
                "compute_standard_errors must be a boolean",
                parameter="compute_standard_errors",
                value=compute_standard_errors,
                expected="bool",
            )
        self.n_quadpts = int(n_quadpts)
        self.compute_standard_errors = bool(compute_standard_errors)
        self.se_method = validate_se_method(se_method)
        self.quadrature = validate_quadrature_rule(quadrature)

    def fit(
        self,
        model: BifactorModel | GradedBifactorModel | TestletModel,
        responses: NDArray[np.int_],
        *,
        start: StartValues = "default",
    ) -> FitResult:
        """Fit a bifactor or testlet model by marginal maximum likelihood.

        Parameters
        ----------
        model : BifactorModel, GradedBifactorModel or TestletModel
            Model to fit in place. Coordinates fixed with
            ``set_free_parameter_masks`` keep their values. A testlet model
            (``TestletModel``, ``BifactorTestletModel`` or
            ``RandomTestletEffectsModel``) estimates its free loadings, its
            common testlet loadings, or the variances of testlets whose
            loadings are fixed.
        responses : ndarray of shape (n_persons, n_items)
            Dichotomous responses, or category codes for a
            ``GradedBifactorModel``, with negative values marking missing
            ones.
        start : {"default", "model"} or mapping, default="default"
            Starting values, as for :meth:`EMEstimator.fit`.

        Returns
        -------
        FitResult
            Fitted model, log-likelihood, standard errors and fit statistics.

        Raises
        ------
        MirtModelError
            If ``model`` is not a built-in ``BifactorModel``,
            ``GradedBifactorModel`` or testlet model.
        MirtDataError
            If an observed response is not 0 or 1, or for a
            ``GradedBifactorModel`` not one of an item's category codes.
        """
        from mirt.estimation._refit import recipe_for
        from mirt.results.fit_result import FitResult

        problem = _problem_for(model)
        start = _validate_start(start)
        responses = self._validate_responses(responses, model.n_items)
        if isinstance(problem, _GradedBifactorProblem):
            problem.validate_responses(cast("GradedBifactorModel", model), responses)
        elif np.any(responses > 1):
            raise MirtDataError(
                "BifactorEMEstimator requires dichotomous 0/1 responses; use a "
                "GradedBifactorModel (bfactor(model='graded')) for graded items",
                n_persons=responses.shape[0],
                n_items=responses.shape[1],
            )
        _apply_starting_values(model, start)
        grid = problem.grid(self.n_quadpts, model, self.quadrature)

        with EMFitContext(responses, compress=True) as context:
            log_likelihood, converged, n_iterations = self._run_em(
                model, grid, context, problem
            )
            model._is_fitted = True
            standard_errors: dict[str, NDArray[np.float64]] = {}
            se_method = covariance = None
            if self.compute_standard_errors:
                se_method = "oakes" if self.se_method == "auto" else self.se_method
                standard_errors, covariance = problem.standard_errors(
                    model, grid, context, se_method
                )

        n_persons = context.n_observations
        n_parameters = model.n_parameters
        return FitResult(
            model=model,
            log_likelihood=log_likelihood,
            n_iterations=n_iterations,
            converged=converged,
            standard_errors=standard_errors,
            aic=self._compute_aic(log_likelihood, n_parameters),
            bic=self._compute_bic(log_likelihood, n_parameters, n_persons),
            n_observations=n_persons,
            n_parameters=n_parameters,
            se_method=se_method,
            vcov=covariance,
            refit_recipe=recipe_for(self),
        )

    def _run_em(
        self,
        model: BifactorModel | GradedBifactorModel | TestletModel,
        grid: _Grid,
        context: EMFitContext,
        problem: _Problem | None = None,
    ) -> tuple[float, bool, int]:
        """Iterate E- and M-steps and return the final log-likelihood and status."""
        if problem is None:
            problem = _BifactorProblem()
        self._convergence_history = []
        previous = -np.inf
        converged = False
        for iteration in range(self.max_iter):
            current, statistics = problem.expected_counts(model, grid, context)
            self._convergence_history.append(current)
            self._log_iteration(iteration, current)
            if self._check_convergence(previous, current):
                converged = True
                break
            previous = current
            problem.m_step(model, grid, *statistics)
        else:
            current, _ = problem.expected_counts(model, grid, context)
            self._convergence_history.append(current)
            converged = self._check_convergence(previous, current)
        if self.verbose and converged:
            print(f"Converged at iteration {iteration}")
        return current, converged, iteration + 1

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_quadpts={self.n_quadpts}, "
            f"max_iter={self.max_iter}, tol={self.tol}, "
            f"quadrature={self.quadrature!r})"
        )


def _require_bifactor_model(model: object) -> None:
    """Raise unless ``model`` is a model this estimator fits.

    Built-in bifactor and testlet models qualify; customized ones do not.
    """
    _problem_for(model)


def _problem_for(model: object) -> _Problem:
    """Return the reduced-grid problem of a built-in bifactor or testlet model.

    Raises
    ------
    MirtModelError
        If ``model`` is neither, or overrides one of their hooks.
    """
    from mirt.models.bifactor import BifactorModel, GradedBifactorModel

    if _is_testlet_model(model):
        return _TestletProblem(cast("TestletModel", model))
    if (
        type(model) not in (BifactorModel, GradedBifactorModel)
        or not uses_builtin_model_hooks(model, likelihood=True)
        or tuple(cast("BifactorModel", model)._parameters) != _PARAMETERS
    ):
        raise MirtModelError(
            "BifactorEMEstimator requires a built-in BifactorModel or testlet "
            "model, or a GradedBifactorModel; use EMEstimator for other or "
            "customized models",
            model_type=type(model).__name__,
        )
    if type(model) is GradedBifactorModel:
        return _GradedBifactorProblem(model)
    return _BifactorProblem()


# Reduced-grid problem of each supported model family.
_Problem: TypeAlias = "_BifactorProblem | _TestletProblem | _GradedBifactorProblem"


class _DichotomousCounts:
    """E-step of the problems whose items are dichotomous on the grid."""

    def coefficients(self, model: Any) -> NDArray[np.float64]:
        raise NotImplementedError

    def expected_counts(
        self, model: Any, grid: _Grid, context: EMFitContext
    ) -> tuple[float, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return the log-likelihood and the expected (correct, observed) counts."""
        curves = _Curves.evaluate(grid, self.coefficients(model))
        current, correct, observed = _expected_counts(grid, curves, context)
        return current, (correct, observed)


class _BifactorProblem(_DichotomousCounts):
    """Coefficients, M-step and errors of a :class:`BifactorModel`."""

    @staticmethod
    def grid(
        n_points: int, model: Any, rule: QuadratureRule = "gauss-hermite"
    ) -> _Grid:
        return _Grid.build(n_points, model, rule)

    def coefficients(self, model: Any) -> NDArray[np.float64]:
        return _coefficients(model)

    @staticmethod
    def m_step(
        model: Any,
        grid: _Grid,
        correct: NDArray[np.float64],
        observed: NDArray[np.float64],
    ) -> None:
        _m_step(model, grid, correct, observed)

    @staticmethod
    def standard_errors(
        model: Any, grid: _Grid, context: EMFitContext, method: str
    ) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64] | None]:
        return _standard_errors(model, grid, context, method)


def _m_step(
    model: BifactorModel,
    grid: _Grid,
    correct: NDArray[np.float64],
    observed: NDArray[np.float64],
) -> None:
    """Maximize every item's expected complete-data log-likelihood.

    All items share the ``(general, specific)`` design, so free items are
    solved jointly by Newton's method. Items with fixed coordinates, or with
    Newton estimates that fail or leave the parameter box, use a bounded
    solve. Items without observed responses keep their values.
    """
    coefficients = _coefficients(model)
    masks = model.free_parameter_masks
    free = np.column_stack([masks[name] for name in _PARAMETERS])
    bounds = np.array([_parameter_bounds(model, name) for name in _PARAMETERS])
    shape = coefficients.shape
    updated = _solve_items(
        grid.design,
        correct,
        observed,
        coefficients,
        free,
        np.broadcast_to(bounds[:, 0], shape),
        np.broadcast_to(bounds[:, 1], shape),
    )
    model.set_parameters(
        **{name: updated[:, column] for column, name in enumerate(_PARAMETERS)}
    )


def _solve_items(
    design: NDArray[np.float64],
    correct: NDArray[np.float64],
    observed: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    free: NDArray[np.bool_],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return every item's maximizing ``(a_g, a_s, d)`` over its free coordinates.

    ``free``, ``lower`` and ``upper`` hold one row per item. Items whose
    coordinates are all free are solved jointly by Newton's method; the
    others, and Newton estimates that fail or leave their box, by a bounded
    solve. Items without observed responses keep their values.
    """
    estimable = np.any(observed > 0.0, axis=1) & np.any(free, axis=1)
    newton = np.flatnonzero(estimable & np.all(free, axis=1))
    bounded = np.flatnonzero(estimable & ~np.all(free, axis=1))
    updated = coefficients.copy()
    if newton.size:
        slopes, intercepts, solved = newton_logistic_items(
            design[:, :2],
            correct[newton],
            observed[newton],
            coefficients[newton, :2],
            coefficients[newton, 2],
        )
        candidate = np.column_stack((slopes, intercepts))
        solved &= np.all(
            (candidate >= lower[newton]) & (candidate <= upper[newton]), axis=1
        )
        updated[newton[solved]] = candidate[solved]
        bounded = np.concatenate((bounded, newton[~solved]))
    for item in bounded:
        updated[item] = _bounded_item_update(
            design,
            correct[item],
            observed[item],
            coefficients[item],
            free[item],
            lower[item],
            upper[item],
        )
    return updated


@dataclass(frozen=True)
class _Information:
    """Item-major information terms; coordinate ``c`` of item ``j`` is ``3j + c``.

    ``information`` is ``None`` when only the score cross-product was needed.
    ``complete`` holds the diagonal of the expected complete-data information.
    """

    information: NDArray[np.float64] | None
    score_crossproduct: NDArray[np.float64]
    complete: NDArray[np.float64]


def _louis_information(
    model: BifactorModel,
    grid: _Grid,
    context: EMFitContext,
    *,
    observed: bool = True,
) -> _Information:
    """Accumulate the exact observed information of a bifactor model.

    See :func:`_louis_terms`.
    """
    return _louis_terms(_coefficients(model), grid, context, observed=observed)


def _louis_terms(
    coefficients: NDArray[np.float64],
    grid: _Grid,
    context: EMFitContext,
    *,
    observed: bool = True,
) -> _Information:
    """Accumulate the exact observed information from reduced posteriors.

    ``coefficients`` holds each item's ``(a_g, a_s, d)``, and the result is
    indexed by these coordinates.

    Louis's identity writes a person's observed information as the posterior
    mean of the complete-data information minus the posterior variance of the
    complete-data score ``S``. Item scores within one specific factor share
    the (general, specific) posterior. Scores of different specific factors
    are conditionally independent given the general factor, so their
    cross-moments need only the conditional score means
    ``m(g) = E[S | g, y]``:

    ``E[S S'] = sum_g p(g) m(g) m(g)'`` off the specific-factor blocks, while
    each diagonal block is the full second moment on its own grid.
    """
    curves = _Curves.evaluate(grid, coefficients)
    n_items = coefficients.shape[0]
    size = 3 * n_items
    n_points = grid.n_points
    design = grid.design
    shape = (n_points, n_points)
    # Derivatives of the clipped log-probabilities vanish where clipping binds.
    residual_correct = np.where(curves.active, 1.0 - curves.probability, 0.0)
    residual_incorrect = np.where(curves.active, -curves.probability, 0.0)
    specific = grid.nodes[None, None, :]

    score_outer = np.zeros((size, size))
    conditional_outer = np.zeros((size, size))
    within = [
        np.zeros((design.shape[0], items.size, items.size)) for items in grid.groups
    ]
    observed_counts = np.zeros((n_items, design.shape[0]))
    largest = max(items.size for items in grid.groups)
    # Per row: the conditionals, two item residual arrays of one factor, and
    # the conditional score means with their weighted copy (2 * Q * 3J).
    block_size = grid.block_size(
        len(grid.groups) + 2 * largest + -(-6 * n_items // n_points)
    )
    for block in _posterior_blocks(grid, curves, context, block_size):
        n = block.stop - block.start
        weights = _row_weights(context.frequencies, block.start, block.stop)
        person = np.zeros((n, n_items, 3))
        means = np.zeros((n, n_points, n_items, 3))
        for factor, items in enumerate(grid.groups):
            correct = block.correct[:, items]
            incorrect = block.observed[:, items] - correct
            # residual[i, q, j]: score factor of item j at node q, (y - P).
            residual = correct[:, None, :] * residual_correct[None, :, items]
            residual += incorrect[:, None, :] * residual_incorrect[None, :, items]
            conditional = block.conditionals[factor]
            grid_residual = residual.reshape(n, *shape, items.size)
            level = np.matmul(conditional[:, :, None, :], grid_residual)[:, :, 0]
            slope = np.matmul((conditional * specific)[:, :, None, :], grid_residual)[
                :, :, 0
            ]
            means[:, :, items, 0] = grid.nodes[None, :, None] * level
            means[:, :, items, 1] = slope
            means[:, :, items, 2] = level
            person[:, items] = np.einsum(
                "ig,igjc->ijc", block.general, means[:, :, items]
            )
            joint = block.joint(factor)
            observed_counts[items] += (
                block.observed[:, items] * weights[:, None]
            ).T @ joint
            if observed:
                scaled = residual * np.sqrt(joint * weights[:, None])[:, :, None]
                within[factor] += np.matmul(
                    scaled.transpose(1, 2, 0), scaled.transpose(1, 0, 2)
                )
        person = person.reshape(n, size)
        score_outer += (person * weights[:, None]).T @ person
        if observed:
            scale = np.sqrt(block.general * weights[:, None])
            flat = (means * scale[:, :, None, None]).reshape(n * n_points, size)
            conditional_outer += flat.T @ flat

    curvature = (
        observed_counts
        * np.where(
            curves.active, curves.probability * (1.0 - curves.probability), 0.0
        ).T
    )
    complete_blocks = np.einsum("jq,qc,qd->jcd", curvature, design, design)
    complete = np.einsum("jcc->jc", complete_blocks).ravel()
    if not observed:
        return _Information(None, score_outer, complete)

    second_moment = conditional_outer
    for factor, items in enumerate(grid.groups):
        index = (3 * items[:, None] + np.arange(3)).ravel()
        moment = np.einsum("qjl,qc,qd->jcld", within[factor], design, design)
        second_moment[np.ix_(index, index)] = moment.reshape(index.size, index.size)
    information = score_outer - second_moment
    for item in range(n_items):
        index = slice(3 * item, 3 * item + 3)
        information[index, index] += complete_blocks[item]
    return _Information((information + information.T) / 2.0, score_outer, complete)


def _standard_errors(
    model: BifactorModel,
    grid: _Grid,
    context: EMFitContext,
    method: str,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64] | None]:
    """Return standard errors and the free-parameter covariance, if any."""
    from mirt.estimation.standard_errors import _flatten_parameters

    _, layouts = _flatten_parameters(model)
    order = np.concatenate(
        [
            3 * layouts[name].free_indices + column
            for column, name in enumerate(_PARAMETERS)
        ]
    )
    terms = _louis_information(
        model, grid, context, observed=method in ("oakes", "sandwich")
    )
    return _errors_from_terms(model, layouts, terms, order, method)


def _errors_from_terms(
    model: BifactorModel | GradedBifactorModel,
    layouts: Mapping[str, Any],
    terms: _Information,
    order: NDArray[np.intp],
    method: str,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64] | None]:
    """Invert item-major information terms for the free parameters.

    ``order`` gives the item-major coordinate of each free parameter in
    ``layouts`` order.
    """
    from mirt.estimation.standard_errors import (
        _coordinates_at_bounds,
        _covariance,
        _se_from_covariance,
        _unflatten_se,
    )

    layouts = dict(layouts)
    if method == "complete_data":
        curvature = terms.complete[order]
        errors = np.full(order.size, np.nan)
        positive = curvature > 0.0
        errors[positive] = 1.0 / np.sqrt(curvature[positive])
        return _unflatten_se(errors, layouts, model), None

    block = np.ix_(order, order)
    information = None if terms.information is None else terms.information[block]
    active = ~_coordinates_at_bounds(
        layouts, lambda name: _parameter_bounds(model, name)
    )
    covariance = _covariance(
        cast(Literal["oakes", "crossprod", "sandwich"], method),
        information,
        terms.score_crossproduct[block],
        active,
    )
    return _se_from_covariance(covariance, layouts, model), covariance


# Stored parameters of a built-in testlet model, in model order.
_TESTLET_PARAMETERS = (
    "discrimination",
    "testlet_loadings",
    "difficulty",
    "testlet_variances",
)
# Testlet hooks whose replacement on an instance would change its curves or
# free coordinates, which the reduced grids compute from the parameters.
_TESTLET_HOOKS = (
    "probability",
    "log_likelihood",
    "log_likelihood_batch",
    "set_parameters",
    "free_parameter_masks",
    "_testlet_slopes",
    "_canonical_parameter_values",
    "_family_loading_mask",
)
_SCALE_MAXITER = 50


def _is_testlet_model(model: object) -> bool:
    """Whether ``model`` is a built-in testlet model with its own hooks."""
    from mirt.models.testlet import (
        BifactorTestletModel,
        RandomTestletEffectsModel,
        TestletModel,
    )

    return (
        type(model) in (TestletModel, BifactorTestletModel, RandomTestletEffectsModel)
        and tuple(getattr(model, "_parameters", ())) == _TESTLET_PARAMETERS
        and not any(name in vars(model) for name in _TESTLET_HOOKS)
    )


def _scale_newton(
    design: NDArray[np.float64],
    correct: NDArray[np.float64],
    observed: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    multipliers: NDArray[np.float64],
    start: float,
    bounds: tuple[float, float],
) -> float:
    """Maximize a testlet's summed expected log-likelihood over one scale ``s``.

    The testlet's items have specific slopes ``multipliers * s`` and fixed
    general slopes and intercepts. Their logits are linear in ``s``, so the
    objective is concave and damped Newton steps within ``bounds`` converge.
    Returns ``start`` unless a step raises the objective.
    """
    offsets = design[:, [0, 2]] @ coefficients[:, [0, 2]].T
    direction = design[:, 1:2] * multipliers[None, :]
    successes = correct.T
    trials = observed.T

    def evaluate(scale: float) -> tuple[float, float, float]:
        logits = offsets + scale * direction
        log_p = -np.logaddexp(0.0, -logits)
        log_q = -np.logaddexp(0.0, logits)
        probability = np.exp(log_p)
        value = float(np.sum(successes * log_p + (trials - successes) * log_q))
        gradient = float(np.sum((successes - trials * probability) * direction))
        curvature = -float(
            np.sum(trials * probability * (1.0 - probability) * direction**2)
        )
        return value, gradient, curvature

    low, high = bounds
    current = float(np.clip(start, low, high))
    start_value, gradient, curvature = evaluate(current)
    best = start_value
    for _ in range(_SCALE_MAXITER):
        if not (np.isfinite(gradient) and curvature < 0.0):
            break
        trial = float(np.clip(current - gradient / curvature, low, high))
        trial_value, trial_gradient, trial_curvature = evaluate(trial)
        # Halve steps that overshoot; the objective is concave.
        while trial_value < best and abs(trial - current) > 1e-12:
            trial = 0.5 * (trial + current)
            trial_value, trial_gradient, trial_curvature = evaluate(trial)
        if trial_value < best:
            break
        converged = abs(trial - current) <= 1e-10 * max(1.0, abs(current))
        current, best = trial, trial_value
        gradient, curvature = trial_gradient, trial_curvature
        if converged:
            break
    return current if best > start_value else float(start)


class _TestletProblem(_DichotomousCounts):
    """Bifactor form of a testlet model and its testlet-level M-steps.

    Item ``j`` of testlet ``t`` has coefficients ``(a_j, d_j * sigma_t,
    -b_j)`` on the (general, specific) grid of its testlet. Standalone items
    have no specific slope and join the first testlet's grid, which leaves
    its posterior unchanged.
    """

    def __init__(self, model: TestletModel) -> None:
        positions = model._testlet_positions
        self.positions = positions
        self.members = tuple(
            np.flatnonzero(positions == position)
            for position in range(model.n_testlets)
        )
        standalone = np.flatnonzero(positions < 0)
        self.groups: tuple[NDArray[np.intp], ...] = (
            (np.sort(np.concatenate((self.members[0], standalone))), *self.members[1:])
            if self.members
            else (standalone,)
        )

    def grid(
        self,
        n_points: int,
        model: TestletModel,
        rule: QuadratureRule = "gauss-hermite",
    ) -> _Grid:
        return _Grid.from_groups(n_points, self.groups, rule)

    def coefficients(self, model: TestletModel) -> NDArray[np.float64]:
        params = model._parameters
        return np.column_stack(
            (params["discrimination"], model._testlet_slopes(), -params["difficulty"])
        )

    def _item_deviations(self, model: TestletModel) -> NDArray[np.float64]:
        """Return each item's testlet standard deviation, zero for standalone."""
        sigma = np.sqrt(model._parameters["testlet_variances"])
        deviations = np.zeros(self.positions.size)
        in_testlet = self.positions >= 0
        deviations[in_testlet] = sigma[self.positions[in_testlet]]
        return deviations

    def m_step(
        self,
        model: TestletModel,
        grid: _Grid,
        correct: NDArray[np.float64],
        observed: NDArray[np.float64],
    ) -> None:
        """Update the items, then each testlet's free common scale."""
        params = model._parameters
        masks = model.free_parameter_masks
        shared_loadings = "testlet_loadings" in model._shared_parameters
        deviations = self._item_deviations(model)
        loading_free = (
            masks["testlet_loadings"] & (deviations > 0)
            if not shared_loadings
            else np.zeros(model.n_items, dtype=np.bool_)
        )
        free = np.column_stack(
            (masks["discrimination"], loading_free, masks["difficulty"])
        )
        slope = _parameter_bounds(model, "discrimination")
        loading = _parameter_bounds(model, "testlet_loadings")
        location = _parameter_bounds(model, "difficulty")
        n_items = model.n_items
        lower = np.column_stack(
            (
                np.full(n_items, slope[0]),
                loading[0] * deviations,
                np.full(n_items, -location[1]),
            )
        )
        upper = np.column_stack(
            (
                np.full(n_items, slope[1]),
                loading[1] * deviations,
                np.full(n_items, -location[0]),
            )
        )
        updated = _solve_items(
            grid.design, correct, observed, self.coefficients(model), free, lower, upper
        )

        loadings = params["testlet_loadings"].copy()
        variances = params["testlet_variances"].copy()
        loadings[loading_free] = updated[loading_free, 1] / deviations[loading_free]
        variance_bounds = np.sqrt(_parameter_bounds(model, "testlet_variances"))
        for position, items in enumerate(self.members):
            # A testlet's variance is free only with its loadings fixed, and
            # constrained loadings are shared only with its variance fixed.
            if masks["testlet_variances"][position]:
                multipliers = loadings[items]
                start = float(np.sqrt(variances[position]))
                box = (float(variance_bounds[0]), float(variance_bounds[1]))
            elif (
                shared_loadings
                and np.any(masks["testlet_loadings"][items])
                and deviations[items[0]] > 0
            ):
                multipliers = deviations[items]
                start = float(loadings[items[0]])
                box = loading
            else:
                continue
            scale = _scale_newton(
                grid.design,
                correct[items],
                observed[items],
                updated[items],
                multipliers,
                start,
                box,
            )
            if masks["testlet_variances"][position]:
                variances[position] = scale**2
            else:
                loadings[items] = scale
        model.set_parameters(
            discrimination=updated[:, 0],
            testlet_loadings=loadings,
            difficulty=-updated[:, 2],
            testlet_variances=variances,
        )

    def _jacobian(
        self, model: TestletModel, layouts: Mapping[str, Any]
    ) -> NDArray[np.float64]:
        """Map the free testlet coordinates to the items' ``(a_g, a_s, d)``.

        Rows follow the item-major bifactor coordinates and columns the free
        coordinates in ``model.parameters`` order. A constrained common
        loading moves the specific slope of every item of its testlet.
        """
        n_items = model.n_items
        sigma = np.sqrt(model._parameters["testlet_variances"])
        loadings = model._parameters["testlet_loadings"]
        shared_loadings = "testlet_loadings" in model._shared_parameters
        columns = []
        for name, layout in layouts.items():
            for flat in np.asarray(layout.free_indices, dtype=np.intp):
                column = np.zeros(3 * n_items)
                if name == "discrimination":
                    column[3 * flat] = 1.0
                elif name == "difficulty":
                    column[3 * flat + 2] = -1.0
                elif name == "testlet_loadings":
                    position = int(self.positions[flat])
                    items = self.members[position] if shared_loadings else flat
                    column[3 * np.atleast_1d(items) + 1] = sigma[position]
                elif sigma[flat] > 0.0:
                    # a_s = d * sqrt(v), so d a_s / d v = d / (2 sigma).
                    items = self.members[flat]
                    column[3 * items + 1] = loadings[items] / (2.0 * sigma[flat])
                columns.append(column)
        if not columns:
            return np.zeros((3 * n_items, 0))
        return np.column_stack(columns)

    def standard_errors(
        self,
        model: TestletModel,
        grid: _Grid,
        context: EMFitContext,
        method: str,
    ) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64] | None]:
        """Return standard errors and the free-parameter covariance, if any.

        The information of the free testlet coordinates is ``J' I J`` for
        the bifactor information ``I`` and the linear map ``J`` of
        :meth:`_jacobian`, exact at the estimates because each free
        coordinate enters the specific slopes linearly given the fixed
        ones.
        """
        from mirt.estimation.standard_errors import (
            _coordinates_at_bounds,
            _covariance,
            _flatten_parameters,
            _se_from_covariance,
            _unflatten_se,
        )

        _, layouts = _flatten_parameters(model)
        jacobian = self._jacobian(model, layouts)
        terms = _louis_terms(
            self.coefficients(model),
            grid,
            context,
            observed=method in ("oakes", "sandwich"),
        )
        if method == "complete_data":
            # Each coordinate moves at most one coefficient of an item, and
            # the complete-data information is block-diagonal in the items.
            curvature = (jacobian**2).T @ terms.complete
            errors = np.full(curvature.size, np.nan)
            positive = curvature > 0.0
            errors[positive] = 1.0 / np.sqrt(curvature[positive])
            return _unflatten_se(errors, layouts, model), None
        information = (
            None
            if terms.information is None
            else jacobian.T @ terms.information @ jacobian
        )
        meat = jacobian.T @ terms.score_crossproduct @ jacobian
        active = ~_coordinates_at_bounds(
            layouts, lambda name: _parameter_bounds(model, name)
        )
        covariance = _covariance(
            cast(Literal["oakes", "crossprod", "sandwich"], method),
            information,
            meat,
            active,
        )
        return _se_from_covariance(covariance, layouts, model), covariance


# Newton settings of the graded M-step and its smallest intercept gap,
# with slack for the roundoff of projected and SLSQP solutions.
_GRADED_MAXITER = 50
_GRADED_TOL = 1e-10
_GRADED_HALVINGS = 30
_GRADED_RIDGE = 1e-12
_GRADED_GAP = THRESHOLD_GAP * (1.0 - 1e-6)


@dataclass(frozen=True)
class _CategoryLayout:
    """Category columns and item coordinates of a graded bifactor model.

    Item ``j`` owns the stacked category columns ``offsets[j]`` to
    ``offsets[j] + sizes[j] - 1`` and the item-major coordinates
    ``first[j] + (0, 1, 2, ...)`` of ``(a_g, a_s, d_1, ..., d_{C-1})``.
    ``columns`` and ``coordinates`` collect them per specific factor, and
    ``thresholds`` groups the items by number of intercepts.
    """

    sizes: NDArray[np.intp]
    offsets: NDArray[np.intp]
    first: NDArray[np.intp]
    groups: tuple[NDArray[np.intp], ...]
    columns: tuple[NDArray[np.intp], ...]
    coordinates: tuple[NDArray[np.intp], ...]
    thresholds: tuple[tuple[int, NDArray[np.intp]], ...]

    @classmethod
    def build(cls, model: GradedBifactorModel) -> _CategoryLayout:
        sizes = np.asarray(model.n_categories, dtype=np.intp)
        offsets = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.intp)
        first = np.concatenate(([0], np.cumsum(sizes + 1)[:-1])).astype(np.intp)
        indices = model._specific_factor_indices
        groups = tuple(
            np.flatnonzero(indices == factor)
            for factor in range(model.n_specific_factors)
        )

        def ranges(starts: NDArray[np.intp], widths: NDArray[np.intp]) -> NDArray:
            return np.concatenate(
                [np.arange(s, s + w) for s, w in zip(starts, widths, strict=True)]
            ).astype(np.intp)

        return cls(
            sizes,
            offsets,
            first,
            groups,
            tuple(ranges(offsets[items], sizes[items]) for items in groups),
            tuple(ranges(first[items], sizes[items] + 1) for items in groups),
            tuple(
                (int(count) - 1, np.flatnonzero(sizes == count))
                for count in np.unique(sizes)
            ),
        )

    @property
    def n_columns(self) -> int:
        return int(self.sizes.sum())

    @property
    def n_coordinates(self) -> int:
        return int(self.sizes.sum() + self.sizes.size)

    def indicators(self, responses: NDArray[np.int_]) -> NDArray[np.float64]:
        """Return one-hot category columns of validated responses; missing is zero."""
        rows, items = np.nonzero(responses >= 0)
        indicators = np.zeros((responses.shape[0], self.n_columns))
        indicators[rows, self.offsets[items] + responses[rows, items]] = 1.0
        return indicators

    def item_counts(
        self, counts: NDArray[np.float64], items: NDArray[np.intp], width: int
    ) -> NDArray[np.float64]:
        """Gather ``(m, width, Q * Q)`` category counts of items of one width."""
        return counts[self.offsets[items, None] + np.arange(width)]


@dataclass(frozen=True)
class _GradedCurves:
    """Clipped category log-probabilities ``(n_columns, Q * Q)`` on the grid.

    ``active`` marks the probabilities that the model's likelihood does not
    clip; derivatives of the clipped log-probabilities vanish elsewhere.
    """

    log_probability: NDArray[np.float64]
    active: NDArray[np.bool_]

    @classmethod
    def evaluate(
        cls, grid: _Grid, model: GradedBifactorModel, layout: _CategoryLayout
    ) -> _GradedCurves:
        # Every specific column holds the specific nodes, so each item is
        # evaluated on its own factor's (general, specific) grid.
        theta = np.empty((grid.design.shape[0], model.n_factors))
        theta[:, 0] = grid.design[:, 0]
        theta[:, 1:] = grid.design[:, 1:2]
        present = np.arange(model.max_categories) < layout.sizes[:, None]
        table = model.probability(theta).transpose(1, 2, 0)[present]
        active = (table > PROB_EPSILON) & (table < 1.0 - PROB_EPSILON)
        return cls(np.log(np.clip(table, PROB_EPSILON, 1.0 - PROB_EPSILON)), active)


@dataclass
class _GradedBlock:
    """Reduced posterior of a block of rows with one-hot category indicators."""

    start: int
    stop: int
    indicators: NDArray[np.float64]
    log_marginal: NDArray[np.float64]
    general: NDArray[np.float64]
    conditionals: list[NDArray[np.float64]]

    def joint(self, factor: int) -> NDArray[np.float64]:
        """Return the ``(n, Q * Q)`` posterior of (general, specific) nodes."""
        joint = self.general[:, :, None] * self.conditionals[factor]
        return joint.reshape(joint.shape[0], -1)


def _graded_posterior_blocks(
    grid: _Grid,
    curves: _GradedCurves,
    layout: _CategoryLayout,
    context: EMFitContext,
    block_size: int,
) -> Iterator[_GradedBlock]:
    """Yield exact reduced posteriors of graded responses by blocks of rows."""
    n_points = grid.n_points
    log_weights = np.tile(grid.log_weights, n_points)
    # One product per specific factor gives log P(y_s | g, k) + log w_k.
    tables = [
        np.vstack((curves.log_probability[columns], log_weights))
        for columns in layout.columns
    ]
    responses = context.responses
    for start in range(0, responses.shape[0], block_size):
        stop = min(start + block_size, responses.shape[0])
        n = stop - start
        indicators = layout.indicators(responses[start:stop])
        ones = np.ones((n, 1))
        log_marginal, general, conditionals = _reduced_posterior(
            grid,
            n,
            (
                (np.hstack((indicators[:, columns], ones)) @ table).reshape(
                    n, n_points, n_points
                )
                for columns, table in zip(layout.columns, tables, strict=True)
            ),
        )
        yield _GradedBlock(start, stop, indicators, log_marginal, general, conditionals)


def _graded_expected_counts(
    grid: _Grid,
    curves: _GradedCurves,
    layout: _CategoryLayout,
    context: EMFitContext,
) -> tuple[float, NDArray[np.float64]]:
    """Run an E-step; return the log-likelihood and ``(n_columns, Q * Q)`` counts."""
    counts = np.zeros((layout.n_columns, grid.design.shape[0]))
    log_likelihood = 0.0
    block_size = grid.block_size(len(layout.groups) + 1)
    for block in _graded_posterior_blocks(grid, curves, layout, context, block_size):
        weights = _row_weights(context.frequencies, block.start, block.stop)
        log_likelihood += float(weights @ block.log_marginal)
        for factor, columns in enumerate(layout.columns):
            counts[columns] += (
                block.indicators[:, columns] * weights[:, None]
            ).T @ block.joint(factor)
    return log_likelihood, counts


@dataclass(frozen=True)
class _GradedTerms:
    """Expected negative log-likelihood of graded items and its derivatives.

    For ``m`` items with ``K`` intercepts, ``loss`` has shape ``(m,)``,
    ``gradient`` ``(m, K + 2)`` and ``hessian`` and ``fisher`` (the exact and
    the expected second derivatives) ``(m, K + 2, K + 2)`` in the
    coordinates ``(a_g, a_s, d_1, ..., d_K)``.
    """

    loss: NDArray[np.float64]
    gradient: NDArray[np.float64]
    hessian: NDArray[np.float64] | None
    fisher: NDArray[np.float64] | None


def _graded_log_probabilities(
    design: NDArray[np.float64], coefficients: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``log P*_k``, ``log(1 - P*_k)`` and ``log P_c`` on the grid.

    With cumulative logits ``z_k = a_g g + a_s s + d_k`` and ``P*_k =
    logistic(z_k)``, the middle categories are evaluated without
    cancellation as ``P_c = P*_c (1 - P*_{c+1}) (1 - exp(d_{c+1} - d_c))``.
    Arrays have shape ``(m, Q * Q, K)`` and ``(m, Q * Q, K + 1)``.
    """
    projection = coefficients[:, :2] @ design[:, :2].T
    logits = projection[:, :, None] + coefficients[:, None, 2:]
    log_upper = -np.logaddexp(0.0, -logits)
    log_lower = -np.logaddexp(0.0, logits)
    n_thresholds = logits.shape[2]
    log_p = np.empty((*projection.shape, n_thresholds + 1))
    log_p[..., 0] = log_lower[..., 0]
    log_p[..., -1] = log_upper[..., -1]
    if n_thresholds > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_gaps = np.log(-np.expm1(coefficients[:, 3:] - coefficients[:, 2:-1]))
        log_p[..., 1:-1] = log_upper[..., :-1] + log_lower[..., 1:] + log_gaps[:, None]
    return log_upper, log_lower, log_p


def _graded_ratios(
    log_upper: NDArray[np.float64],
    log_lower: NDArray[np.float64],
    log_p: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``w_k``, ``w_k / P_k`` and ``w_k / P_{k-1}`` for ``w = P* (1 - P*)``.

    ``d log P_k / d z_k = w_k / P_k`` and ``d log P_{k-1} / d z_k =
    -w_k / P_{k-1}``; both ratios stay bounded where the curves underflow.
    """
    log_slope = log_upper + log_lower
    return (
        np.exp(log_slope),
        np.exp(log_slope - log_p[..., 1:]),
        np.exp(log_slope - log_p[..., :-1]),
    )


def _parameter_matrix(
    design: NDArray[np.float64],
    diagonal: NDArray[np.float64],
    off_diagonal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map tridiagonal logit-space matrices on the grid to the coordinates.

    ``diagonal`` ``(m, Q * Q, K)`` and ``off_diagonal`` ``(m, Q * Q, K - 1)``
    hold each node's matrix in the cumulative logits ``z``, whose Jacobian
    in ``(a_g, a_s, d_1, ..., d_K)`` is ``[g 1, s 1, I]``.
    """
    n_items, _, n_thresholds = diagonal.shape
    rows = diagonal.copy()
    rows[..., :-1] += off_diagonal
    rows[..., 1:] += off_diagonal
    total = rows.sum(axis=2)
    general, specific = design[:, 0], design[:, 1]
    matrix = np.zeros((n_items, n_thresholds + 2, n_thresholds + 2))
    matrix[:, 0, 0] = total @ general**2
    matrix[:, 0, 1] = matrix[:, 1, 0] = total @ (general * specific)
    matrix[:, 1, 1] = total @ specific**2
    matrix[:, 0, 2:] = matrix[:, 2:, 0] = np.einsum("mqk,q->mk", rows, general)
    matrix[:, 1, 2:] = matrix[:, 2:, 1] = np.einsum("mqk,q->mk", rows, specific)
    index = np.arange(2, n_thresholds + 2)
    matrix[:, index, index] = diagonal.sum(axis=1)
    off = off_diagonal.sum(axis=1)
    matrix[:, index[:-1], index[1:]] = off
    matrix[:, index[1:], index[:-1]] = off
    return matrix


def _graded_loss(
    design: NDArray[np.float64],
    counts: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return each item's negative expected log-likelihood ``(m,)``.

    ``counts`` has shape ``(m, K + 1, Q * Q)``; the intercepts must decrease.
    """
    _, _, log_p = _graded_log_probabilities(design, coefficients)
    return -np.einsum("mqc,mcq->m", log_p, counts)


def _graded_terms(
    design: NDArray[np.float64],
    counts: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    *,
    second: bool = True,
) -> _GradedTerms:
    """Return the expected loss of graded items and its derivatives.

    The expected log-likelihood is ``sum_q sum_c r_qc log P_c(q)`` for the
    category counts ``r`` of ``counts`` ``(m, K + 1, Q * Q)``. In the
    cumulative logits its gradient is ``r_k a_k - r_{k-1} b_k`` with ``a_k =
    w_k / P_k`` and ``b_k = w_k / P_{k-1}``, and its Hessian is tridiagonal:
    ``r_k a_k (e_k - a_k) - r_{k-1} b_k (e_k + b_k)`` on the diagonal, with
    ``e_k = 1 - 2 P*_k``, and ``r_k a_k b_{k+1}`` beside it. Replacing ``r``
    by its expectation ``n P`` gives the Fisher matrix with diagonal
    ``n w_k (a_k + b_k)`` and off-diagonal ``-n w_k b_{k+1}``.
    """
    log_upper, log_lower, log_p = _graded_log_probabilities(design, coefficients)
    weight, upper, lower = _graded_ratios(log_upper, log_lower, log_p)
    r = counts.transpose(0, 2, 1)
    loss = -np.einsum("mqc,mqc->m", r, log_p)
    score = r[..., 1:] * upper - r[..., :-1] * lower
    totals = score.sum(axis=2)
    gradient = -np.column_stack(
        (totals @ design[:, 0], totals @ design[:, 1], score.sum(axis=1))
    )
    if not second:
        return _GradedTerms(loss, gradient, None, None)
    curvature = np.exp(log_lower) - np.exp(log_upper)
    diagonal = r[..., :-1] * lower * (curvature + lower) - r[..., 1:] * upper * (
        curvature - upper
    )
    off_diagonal = -r[..., 1:-1] * upper[..., :-1] * lower[..., 1:]
    trials = r.sum(axis=2, keepdims=True)
    fisher_diagonal = trials * weight * (upper + lower)
    fisher_off = -trials * weight[..., :-1] * lower[..., 1:]
    return _GradedTerms(
        loss,
        gradient,
        _parameter_matrix(design, diagonal, off_diagonal),
        _parameter_matrix(design, fisher_diagonal, fisher_off),
    )


def _graded_scores(
    design: NDArray[np.float64], coefficients: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return complete-data scores ``(m, K + 1, Q * Q, K + 2)`` of each category.

    The score of category ``c`` in ``(a_g, a_s, d)`` is ``(g u, s u, v)``
    for its logit gradient ``v``, which is ``w_c / P_c`` at ``z_c`` and
    ``-w_{c+1} / P_c`` at ``z_{c+1}``, and ``u = sum(v)``.
    """
    _, upper, lower = _graded_ratios(*_graded_log_probabilities(design, coefficients))
    n_items, n_nodes, n_thresholds = upper.shape
    index = np.arange(n_thresholds)
    logit_scores = np.zeros((n_items, n_thresholds + 1, n_nodes, n_thresholds))
    logit_scores[:, index + 1, :, index] = upper.transpose(2, 0, 1)
    logit_scores[:, index, :, index] = -lower.transpose(2, 0, 1)
    total = logit_scores.sum(axis=3, keepdims=True)
    return np.concatenate(
        (
            total * design[:, 0, None],
            total * design[:, 1, None],
            logit_scores,
        ),
        axis=3,
    )


def _graded_feasible(
    coefficients: NDArray[np.float64],
    free: NDArray[np.bool_],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Whether free coordinates lie in their box and intercepts decrease."""
    inside = np.all(~free | ((coefficients >= lower) & (coefficients <= upper)), axis=1)
    gaps = coefficients[:, 2:-1] - coefficients[:, 3:]
    return inside & np.all(gaps >= _GRADED_GAP, axis=1)


def _project_graded_item(
    values: NDArray[np.float64],
    free: NDArray[np.bool_],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    order: GradedOrder | None,
) -> NDArray[np.float64]:
    """Return the nearest point of the box with decreasing intercepts.

    ``order`` is the item's :func:`~mirt.estimation._graded_order.graded_order`
    in its free-coordinate layout, or ``None`` without movable intercept pairs.
    """
    projected = values.copy()
    projected[free] = np.clip(values[free], lower[free], upper[free])
    if order is not None:
        projected[free] = order.project(projected[free], lower[free], upper[free])
    return projected


def _newton_graded_items(
    design: NDArray[np.float64],
    counts: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    free: NDArray[np.bool_],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Maximize graded items' expected log-likelihoods by batched Newton steps.

    Steps use the exact Hessian where it is positive definite and Fisher
    scoring elsewhere, and are halved until they stay in the box, keep the
    intercepts decreasing and do not increase the loss. Fixed coordinates
    get identity rows and no gradient. Starting points must be feasible.
    Returns the estimates and whether each item reached the step tolerance.
    """
    coefficients = coefficients.copy()
    n_items, size = coefficients.shape
    diagonal = np.arange(size)
    loss = _graded_loss(design, counts, coefficients)
    active = np.ones(n_items, dtype=np.bool_)
    converged = np.zeros(n_items, dtype=np.bool_)
    for _ in range(_GRADED_MAXITER):
        items = np.flatnonzero(active)
        if not items.size:
            break
        terms = _graded_terms(design, counts[items], coefficients[items])
        assert terms.hessian is not None and terms.fisher is not None
        item_free = free[items]
        pairs = item_free[:, :, None] & item_free[:, None, :]
        gradient = np.where(item_free, terms.gradient, 0.0)
        hessian = np.where(pairs, terms.hessian, 0.0)
        fisher = np.where(pairs, terms.fisher, 0.0)
        hessian[:, diagonal, diagonal] += ~item_free
        fisher[:, diagonal, diagonal] += ~item_free
        with np.errstate(invalid="ignore"):
            definite = np.linalg.eigvalsh(hessian)[:, 0] > 0.0
        hessian = np.where(definite[:, None, None], hessian, fisher)
        hessian[:, diagonal, diagonal] += _GRADED_RIDGE * (
            1.0 + hessian[:, diagonal, diagonal]
        )
        try:
            step = -np.linalg.solve(hessian, gradient[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        with np.errstate(invalid="ignore"):
            small = np.max(np.abs(step), axis=1) <= _GRADED_TOL

        start = coefficients[items]
        base_loss = loss[items]
        slack = 1e-12 * np.maximum(1.0, np.abs(base_loss))
        scale = np.ones(items.size)
        pending = np.ones(items.size, dtype=np.bool_)
        for _ in range(_GRADED_HALVINGS):
            rows = np.flatnonzero(pending)
            if not rows.size:
                break
            selected = items[rows]
            trial = start[rows] + scale[rows, None] * step[rows]
            feasible = _graded_feasible(
                trial, free[selected], lower[selected], upper[selected]
            )
            trial_loss = np.full(rows.size, np.inf)
            if np.any(feasible):
                trial_loss[feasible] = _graded_loss(
                    design, counts[selected[feasible]], trial[feasible]
                )
            with np.errstate(invalid="ignore"):
                accept = trial_loss <= base_loss[rows] + slack[rows]
            accepted = selected[accept]
            coefficients[accepted] = trial[accept]
            loss[accepted] = trial_loss[accept]
            pending[rows[accept]] = False
            scale[rows[~accept]] *= 0.5

        # A tiny full step marks a stationary point even when roundoff
        # prevents any further decrease in the objective.
        converged[items[small]] = True
        active[items[small | pending]] = False
    return coefficients, converged


def _bounded_graded_update(
    design: NDArray[np.float64],
    counts: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    free: NDArray[np.bool_],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    order: GradedOrder | None,
) -> NDArray[np.float64]:
    """Maximize one graded item over its box with decreasing intercepts.

    SLSQP starts from the feasible ``coefficients``; its solution is
    projected onto the feasible set and kept only if it lowers the loss.
    """
    counts = counts[None]
    constraints = () if order is None else (order.constraint,)

    def objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        trial = coefficients.copy()
        trial[free] = params
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = _graded_terms(design, counts, trial[None], second=False)
        value = float(terms.loss[0])
        if not np.isfinite(value):
            return np.inf, np.zeros(params.size)
        return value, terms.gradient[0][free]

    result = minimize(
        objective,
        coefficients[free],
        jac=True,
        method="SLSQP",
        bounds=list(zip(lower[free], upper[free], strict=True)),
        constraints=constraints,
        options={"maxiter": _ITEM_MAXITER, "ftol": _ITEM_FTOL},
    )
    candidate = coefficients.copy()
    candidate[free] = result.x
    candidate = _project_graded_item(candidate, free, lower, upper, order)
    current = _graded_loss(design, counts, coefficients[None])[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        value = _graded_loss(design, counts, candidate[None])[0]
    return candidate if np.isfinite(value) and value <= current else coefficients


def _solve_graded_items(
    design: NDArray[np.float64],
    counts: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    free: NDArray[np.bool_],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    model: GradedBifactorModel,
    items: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Return the maximizing ``(a_g, a_s, d_1, ..., d_K)`` of graded items.

    All arrays hold one row per item of ``items``, which share ``K`` and
    whose current values and free masks ``model`` holds. Disordered or
    out-of-box starting points are first projected onto the feasible set, as
    graded estimators order starting thresholds. Items whose Newton
    iterations do not converge, typically at a bound, use a bounded solve.
    Items without observed responses or free coordinates keep their values.
    """

    def order(row: int) -> GradedOrder | None:
        return graded_order(model, int(items[row]), int(np.count_nonzero(free[row])))

    updated = coefficients.copy()
    rows = np.flatnonzero(np.any(counts > 0.0, axis=(1, 2)) & np.any(free, axis=1))
    if not rows.size:
        return updated
    start = coefficients[rows].copy()
    infeasible = ~_graded_feasible(start, free[rows], lower[rows], upper[rows])
    for index in np.flatnonzero(infeasible):
        row = rows[index]
        start[index] = _project_graded_item(
            start[index], free[row], lower[row], upper[row], order(row)
        )
    solved, converged = _newton_graded_items(
        design, counts[rows], start, free[rows], lower[rows], upper[rows]
    )
    updated[rows] = solved
    for index in np.flatnonzero(~converged):
        row = rows[index]
        updated[row] = _bounded_graded_update(
            design,
            counts[row],
            solved[index],
            free[row],
            lower[row],
            upper[row],
            order(row),
        )
    return updated


class _GradedBifactorProblem:
    """E-step, M-step and errors of a :class:`GradedBifactorModel`.

    Item ``j`` has coefficients ``(a_g, a_s, d_1, ..., d_{C_j - 1})`` on the
    (general, specific) grid of its factor and contributes expected counts
    of each of its categories.
    """

    def __init__(self, model: GradedBifactorModel) -> None:
        self.layout = _CategoryLayout.build(model)

    @staticmethod
    def validate_responses(
        model: GradedBifactorModel, responses: NDArray[np.int_]
    ) -> None:
        """Raise ``MirtDataError`` unless observed codes are item categories."""
        model._validate_polytomous_responses(responses)

    @staticmethod
    def grid(
        n_points: int,
        model: GradedBifactorModel,
        rule: QuadratureRule = "gauss-hermite",
    ) -> _Grid:
        return _Grid.build(n_points, model, rule)

    def expected_counts(
        self, model: GradedBifactorModel, grid: _Grid, context: EMFitContext
    ) -> tuple[float, tuple[NDArray[np.float64]]]:
        """Return the log-likelihood and the expected category counts."""
        curves = _GradedCurves.evaluate(grid, model, self.layout)
        current, counts = _graded_expected_counts(grid, curves, self.layout, context)
        return current, (counts,)

    def _item_problems(
        self, model: GradedBifactorModel
    ) -> Iterator[
        tuple[
            int,
            NDArray[np.intp],
            NDArray[np.float64],
            NDArray[np.bool_],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ]:
        """Yield each width's items, coefficients, free masks and boxes."""
        params = model._parameters
        masks = model.free_parameter_masks
        bounds = np.array([_parameter_bounds(model, name) for name in _PARAMETERS])
        for n_thresholds, items in self.layout.thresholds:
            coefficients = np.column_stack(
                (
                    params["general_loadings"][items],
                    params["specific_loadings"][items],
                    params["intercepts"][items, :n_thresholds],
                )
            )
            free = np.column_stack(
                (
                    masks["general_loadings"][items],
                    masks["specific_loadings"][items],
                    masks["intercepts"][items, :n_thresholds],
                )
            )
            column = np.r_[0, 1, np.full(n_thresholds, 2)]
            shape = coefficients.shape
            yield (
                n_thresholds,
                items,
                coefficients,
                free,
                np.broadcast_to(bounds[column, 0], shape),
                np.broadcast_to(bounds[column, 1], shape),
            )

    def m_step(
        self,
        model: GradedBifactorModel,
        grid: _Grid,
        counts: NDArray[np.float64],
    ) -> None:
        """Maximize every item's expected complete-data log-likelihood."""
        params = model._parameters
        general = params["general_loadings"].copy()
        specific = params["specific_loadings"].copy()
        intercepts = params["intercepts"].copy()
        for entry in self._item_problems(model):
            n_thresholds, items, coefficients, free, lower, upper = entry
            updated = _solve_graded_items(
                grid.design,
                self.layout.item_counts(counts, items, n_thresholds + 1),
                coefficients,
                free,
                lower,
                upper,
                model,
                items,
            )
            general[items] = updated[:, 0]
            specific[items] = updated[:, 1]
            intercepts[items, :n_thresholds] = updated[:, 2:]
        model.set_parameters(
            general_loadings=general,
            specific_loadings=specific,
            intercepts=intercepts,
        )

    def standard_errors(
        self,
        model: GradedBifactorModel,
        grid: _Grid,
        context: EMFitContext,
        method: str,
    ) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64] | None]:
        """Return standard errors and the free-parameter covariance, if any."""
        from mirt.estimation.standard_errors import _flatten_parameters

        _, layouts = _flatten_parameters(model)
        first = self.layout.first
        width = model.max_categories - 1
        flat = layouts["intercepts"].free_indices
        order = np.concatenate(
            (
                first[layouts["general_loadings"].free_indices],
                first[layouts["specific_loadings"].free_indices] + 1,
                first[flat // width] + 2 + flat % width,
            )
        ).astype(np.intp)
        terms = self.louis_terms(
            model, grid, context, observed=method in ("oakes", "sandwich")
        )
        return _errors_from_terms(model, layouts, terms, order, method)

    def louis_terms(
        self,
        model: GradedBifactorModel,
        grid: _Grid,
        context: EMFitContext,
        *,
        observed: bool = True,
    ) -> _Information:
        """Accumulate the exact observed information from reduced posteriors.

        Coordinates are item-major, ``first[j] + (0, 1, 2, ...)`` for
        ``(a_g, a_s, d_1, ...)`` of item ``j``. As for dichotomous items
        (:func:`_louis_terms`), scores of different specific factors meet
        only through their conditional means given the general node. The
        complete-data information is the exact negative Hessian of the
        expected log-likelihood at the final expected category counts.
        Derivatives vanish where the likelihood clips a probability.
        """
        layout = self.layout
        design = grid.design
        n_points = grid.n_points
        n_nodes = design.shape[0]
        size = layout.n_coordinates
        curves = _GradedCurves.evaluate(grid, model, layout)
        params = model._parameters
        # Category score tables with a zero row for a missing response.
        tables: list[NDArray[np.float64]] = [np.empty(0)] * model.n_items
        coefficients: dict[int, NDArray[np.float64]] = {}
        for n_thresholds, items in layout.thresholds:
            item_coefficients = np.column_stack(
                (
                    params["general_loadings"][items],
                    params["specific_loadings"][items],
                    params["intercepts"][items, :n_thresholds],
                )
            )
            coefficients[n_thresholds] = item_coefficients
            scores = _graded_scores(design, item_coefficients)
            active = layout.item_counts(curves.active, items, n_thresholds + 1)
            scores *= active[..., None]
            for position, item in enumerate(items):
                tables[item] = np.concatenate(
                    (scores[position], np.zeros((1, n_nodes, n_thresholds + 2)))
                )
        # Positions of each item's coordinates within its factor's block.
        local = [
            [
                (int(item), slice(start, start + int(layout.sizes[item]) + 1))
                for item, start in zip(
                    items,
                    np.concatenate(([0], np.cumsum(layout.sizes[items] + 1)[:-1])),
                    strict=True,
                )
            ]
            for items in layout.groups
        ]

        score_outer = np.zeros((size, size))
        conditional_outer = np.zeros((size, size))
        within = [np.zeros((coords.size, coords.size)) for coords in layout.coordinates]
        counts = np.zeros((layout.n_columns, n_nodes))
        largest = max(coords.size for coords in layout.coordinates)
        # Per row: the conditionals, a factor's scores and their weighted
        # copy, and the conditional score means with their weighted copy.
        block_size = grid.block_size(
            len(layout.groups) + 2 * largest + 1 + -(-2 * size // n_points)
        )
        responses = context.responses
        for block in _graded_posterior_blocks(
            grid, curves, layout, context, block_size
        ):
            n = block.stop - block.start
            weights = _row_weights(context.frequencies, block.start, block.stop)
            rows = responses[block.start : block.stop]
            person = np.zeros((n, size))
            means = np.zeros((n, n_points, size))
            for factor, coords in enumerate(layout.coordinates):
                scores = np.empty((n, n_nodes, coords.size))
                for item, position in local[factor]:
                    codes = np.where(
                        rows[:, item] >= 0, rows[:, item], layout.sizes[item]
                    )
                    scores[:, :, position] = tables[item][codes]
                factor_means = np.einsum(
                    "igk,igkp->igp",
                    block.conditionals[factor],
                    scores.reshape(n, n_points, n_points, coords.size),
                )
                means[:, :, coords] = factor_means
                person[:, coords] = np.einsum("ig,igp->ip", block.general, factor_means)
                joint = block.joint(factor)
                columns = layout.columns[factor]
                counts[columns] += (
                    block.indicators[:, columns] * weights[:, None]
                ).T @ joint
                if observed:
                    scores *= np.sqrt(joint * weights[:, None])[:, :, None]
                    flat = scores.reshape(-1, coords.size)
                    within[factor] += flat.T @ flat
            score_outer += (person * weights[:, None]).T @ person
            if observed:
                scale = np.sqrt(block.general * weights[:, None])
                flat = (means * scale[:, :, None]).reshape(n * n_points, size)
                conditional_outer += flat.T @ flat

        complete_blocks = np.zeros((size, size))
        counts *= curves.active
        for n_thresholds, items in layout.thresholds:
            terms = _graded_terms(
                design,
                layout.item_counts(counts, items, n_thresholds + 1),
                coefficients[n_thresholds],
            )
            assert terms.hessian is not None
            for position, item in enumerate(items):
                index = layout.first[item] + np.arange(n_thresholds + 2)
                complete_blocks[np.ix_(index, index)] = terms.hessian[position]
        complete = np.diag(complete_blocks).copy()
        if not observed:
            return _Information(None, score_outer, complete)

        second_moment = conditional_outer
        for factor, coords in enumerate(layout.coordinates):
            second_moment[np.ix_(coords, coords)] = within[factor]
        information = score_outer - second_moment + complete_blocks
        return _Information((information + information.T) / 2.0, score_outer, complete)


def _is_graded_bifactor_model(model: object) -> bool:
    """Whether ``model`` is a built-in ``GradedBifactorModel`` with its hooks."""
    from mirt.models.bifactor import GradedBifactorModel

    if type(model) is not GradedBifactorModel:
        return False
    try:
        return isinstance(_problem_for(model), _GradedBifactorProblem)
    except MirtModelError:
        return False


def reduced_testlet_estimator(
    estimator: EMEstimator,
    model: object,
    prior_mean: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
) -> BifactorEMEstimator | None:
    """Return the reduced-grid estimator of a model that ``estimator`` fits.

    A plain :class:`~mirt.estimation.em.EMEstimator` fit of a built-in
    testlet model or :class:`~mirt.models.GradedBifactorModel` with a fixed
    standard-normal latent density, no item priors, equality constraints or
    acceleration, automatic or Gauss-Hermite quadrature and the default
    probability clipping is the exact bifactor problem of
    :class:`BifactorEMEstimator`, whose one (general, specific) grid per
    testlet or specific factor replaces the product grid over all factors.
    It gets ``estimator``'s ``n_quadpts`` (21 when unset), iteration and
    standard-error settings.

    Returns
    -------
    BifactorEMEstimator or None
        The estimator, or None when an option needs the product grid.
    """
    from mirt.estimation.em import EMEstimator

    if type(estimator) is not EMEstimator or not (
        _is_testlet_model(model) or _is_graded_bifactor_model(model)
    ):
        return None
    n_factors = cast("TestletModel", model).n_factors
    standard = (
        prior_mean is None
        or np.array_equal(np.asarray(prior_mean), np.zeros(n_factors))
    ) and (
        prior_cov is None or np.array_equal(np.asarray(prior_cov), np.eye(n_factors))
    )
    if (
        not standard
        or estimator._latent_density_spec not in (None, "gaussian", "normal")
        or estimator.item_priors is not None
        or estimator.constraints
        or estimator.accelerate != "none"
        or estimator.quadrature == "rectangular"
        or estimator.prob_epsilon != PROB_EPSILON
    ):
        return None
    return BifactorEMEstimator(
        n_quadpts=21 if estimator.n_quadpts is None else estimator.n_quadpts,
        max_iter=estimator.max_iter,
        tol=estimator.tol,
        verbose=estimator.verbose,
        compute_standard_errors=estimator.compute_standard_errors,
        se_method=estimator.se_method,
    )


# R mirt's itemtype names and this package's family names of bfactor items.
_BFACTOR_MODELS = {"2PL": "2PL", "graded": "graded", "GRM": "graded"}


def bfactor(
    data: NDArray[np.int_] | Any,
    specific_factors: Sequence[int] | NDArray[np.int_],
    *,
    model: Literal["2PL", "graded", "GRM"] | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    item_names: list[str] | None = None,
    compute_standard_errors: bool = True,
    se_method: StandardErrorMethod = "auto",
    start_values: Mapping[str, ArrayLike] | None = None,
    fixed: Mapping[str, ArrayLike] | None = None,
    quadrature: QuadratureRule = "gauss-hermite",
) -> FitResult:
    """Fit a full-information bifactor model, like R's ``mirt::bfactor``.

    Every item loads on one general factor and on one specific factor. The
    model is fitted by :class:`BifactorEMEstimator`, whose dimension
    reduction integrates over one two-dimensional grid per specific factor,
    so the default 21 points per factor stay practical for any number of
    specific factors.

    Parameters
    ----------
    data : ndarray or DataFrame of shape (n_persons, n_items)
        Responses coded ``0, 1, ...``: dichotomous for ``model="2PL"`` and
        ordered categories for ``model="graded"``. Negative values and
        ``NaN`` mark missing ones.
    specific_factors : sequence of int
        Specific-factor label of each item. Labels may be any non-negative
        integers; they are kept for reporting.
    model : {"2PL", "graded", "GRM"}, optional
        Item type, R's ``itemtype``. ``"2PL"`` fits a dichotomous
        :class:`~mirt.models.BifactorModel` and ``"graded"`` (or ``"GRM"``)
        a :class:`~mirt.models.GradedBifactorModel` with
        ``max(response) + 1`` categories per item, in which two-category
        items are 2PL items. By default, like R, dichotomous data get
        ``"2PL"`` and data with a code above 1 ``"graded"``.
    n_quadpts : int, default=21
        Quadrature points per factor.
    max_iter : int, default=500
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence tolerance for the change in log-likelihood.
    verbose : bool, default=False
        Print iteration progress.
    item_names : list of str, optional
        Item names. Defaults to unique DataFrame column names, else
        ``Item_1``, ``Item_2`` and so on.
    compute_standard_errors : bool, default=True
        Whether to compute standard errors.
    se_method : {"auto", "oakes", "crossprod", "sandwich", "complete_data"}, \
default="auto"
        Standard-error estimator; ``"auto"`` uses the exact observed
        information (``"oakes"``). See :class:`BifactorEMEstimator`.
    start_values : mapping of str to array_like, optional
        Starting values for ``"general_loadings"``, ``"specific_loadings"``
        or ``"intercepts"`` (of shape ``(n_items, max_categories - 1)`` for
        graded items). They also set the values of fixed coordinates.
    fixed : mapping of str to bool or array of bool, optional
        Coordinates to hold at their starting values. Fixing an item's
        specific loading at zero makes it load on the general factor only.
    quadrature : {"gauss-hermite", "rectangular", "auto"}, default="gauss-hermite"
        Rule of the grids. ``quadrature="rectangular", n_quadpts=31`` is the
        grid of R's ``bfactor``.

    Returns
    -------
    FitResult
        Fitted :class:`~mirt.models.BifactorModel` or
        :class:`~mirt.models.GradedBifactorModel` and fit statistics.

    Raises
    ------
    MirtValidationError
        If ``model`` is not an accepted item type, or ``specific_factors``
        does not label every item.
    MirtDataError
        If ``model="2PL"`` gets a response above 1, or a graded item has
        only one observed category.

    Examples
    --------
    >>> import numpy as np
    >>> import mirt
    >>> rng = np.random.default_rng(0)
    >>> factors = np.repeat([0, 1, 2], 4)
    >>> theta = rng.standard_normal((500, 4))
    >>> logits = 1.2 * theta[:, :1] + 0.8 * theta[:, 1 + factors]
    >>> data = (rng.random(logits.shape) < 1 / (1 + np.exp(-logits))).astype(int)
    >>> result = mirt.bfactor(data, factors)
    >>> result.model.n_factors
    4

    Graded (Likert) items, as R's ``bfactor(data, model, itemtype="graded")``:

    >>> cuts = np.array([-1.0, 0.0, 1.0])
    >>> likert = (logits[..., None] + rng.logistic(size=logits.shape)[..., None]
    ...           > cuts).sum(axis=-1)
    >>> graded = mirt.bfactor(likert, factors, model="graded")
    >>> graded.model.n_categories[:3], graded.model.intercepts.shape
    ([4, 4, 4], (12, 3))
    """
    from mirt.estimation.base import _free_masks_from_fixed
    from mirt.models._factory import check_observed_categories, resolve_category_counts
    from mirt.models.bifactor import BifactorModel, GradedBifactorModel
    from mirt.utils.data import response_column_names, validate_responses

    if model is not None and (
        not isinstance(model, str) or model not in _BFACTOR_MODELS
    ):
        raise MirtValidationError(
            "model must be '2PL', 'graded' or 'GRM'",
            parameter="model",
            value=model,
            expected="'2PL', 'graded', 'GRM' or None",
        )
    if item_names is None:
        item_names = response_column_names(data)
    responses = validate_responses(data)
    n_items = responses.shape[1]
    if item_names is None:
        item_names = [f"Item_{index + 1}" for index in range(n_items)]
    family = (
        _BFACTOR_MODELS[model]
        if model is not None
        else ("graded" if np.any(responses > 1) else "2PL")
    )
    irt_model: BifactorModel | GradedBifactorModel
    if family == "graded":
        check_observed_categories(responses, item_names)
    try:
        if family == "graded":
            n_categories = resolve_category_counts(n_items, None, responses)
            irt_model = GradedBifactorModel(
                n_items,
                specific_factors,
                cast("list[int]", n_categories),
                item_names=item_names,
            )
        else:
            irt_model = BifactorModel(n_items, specific_factors, item_names=item_names)
    except MirtValidationError:
        raise
    except ValueError as exc:
        raise MirtValidationError(
            str(exc), parameter="specific_factors", value=specific_factors
        ) from exc
    if fixed is not None:
        irt_model.set_free_parameter_masks(_free_masks_from_fixed(irt_model, fixed))
    estimator = BifactorEMEstimator(
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        compute_standard_errors=compute_standard_errors,
        se_method=se_method,
        quadrature=quadrature,
    )
    return estimator.fit(
        irt_model,
        responses,
        start="default" if start_values is None else start_values,
    )
