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
E-step falls from ``O(N J Q^(1 + S))`` to ``O(N J Q^2)``.

References
----------
Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
    analysis. Psychometrika, 57(3), 423-436.
Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information item
    bifactor analysis. Psychological Methods, 16(3), 221-248.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit

from mirt._model_defaults import uses_builtin_model_hooks
from mirt.constants import PROB_EPSILON
from mirt.estimation._em_context import EMFitContext
from mirt.estimation._logistic_newton import newton_logistic_items
from mirt.estimation.base import (
    BaseEstimator,
    StartValues,
    _apply_starting_values,
    _parameter_bounds,
    _validate_start,
)
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.standard_errors import StandardErrorMethod, validate_se_method
from mirt.exceptions import MirtDataError, MirtModelError, MirtValidationError

if TYPE_CHECKING:
    from mirt.models.bifactor import BifactorModel
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
    an item's logits on the grid are ``design @ (a_g, a_s, d)``.
    """

    nodes: NDArray[np.float64]
    log_weights: NDArray[np.float64]
    design: NDArray[np.float64]
    groups: tuple[NDArray[np.intp], ...]

    @classmethod
    def build(cls, n_points: int, model: BifactorModel) -> _Grid:
        quadrature = GaussHermiteQuadrature(n_points=n_points, n_dimensions=1)
        nodes = quadrature.nodes.ravel()
        weights = quadrature.weights
        general, specific = np.meshgrid(nodes, nodes, indexing="ij")
        design = np.column_stack(
            (general.ravel(), specific.ravel(), np.ones(n_points * n_points))
        )
        indices = model._specific_factor_indices
        groups = tuple(
            np.flatnonzero(indices == factor)
            for factor in range(model.n_specific_factors)
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
        log_general = np.tile(grid.log_weights, (n, 1))
        conditionals = []
        for items, table in zip(grid.groups, tables, strict=True):
            columns = np.hstack((correct[:, items], observed[:, items], ones))
            log_joint = (columns @ table).reshape(n, n_points, n_points)
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
        log_marginal = (shift + np.log(total))[:, 0]
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

    The estimator fits a dichotomous :class:`~mirt.models.BifactorModel` with
    independent standard-normal factors by marginal maximum likelihood. The
    Gibbons-Hedeker dimension reduction integrates each specific factor
    jointly with the general factor only, which reproduces the
    ``(1 + S)``-dimensional product-grid likelihood of :class:`EMEstimator`
    exactly at a cost that grows linearly rather than exponentially in the
    number of specific factors ``S``. Default settings are therefore usable
    for any number of specific factors.

    Parameters
    ----------
    n_quadpts : int, default=21
        Gauss-Hermite quadrature points per factor. Each specific factor uses
        a grid of ``n_quadpts ** 2`` (general, specific) nodes.
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

    Notes
    -----
    Every M-step solves all items without fixed coordinates jointly by
    Newton's method on the expected counts. Items with fixed coordinates, or
    whose Newton estimates leave the parameter box, use a bounded
    quasi-Newton solve. An item whose specific loading is fixed at zero loads on the
    general factor only. Responses must be dichotomous, with negative codes
    for missing responses. Item priors, latent density estimation and
    correlated factors are not supported; use :class:`EMEstimator` for them.

    References
    ----------
    Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
    analysis. *Psychometrika*, 57(3), 423-436.

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

    def fit(
        self,
        model: BifactorModel,
        responses: NDArray[np.int_],
        *,
        start: StartValues = "default",
    ) -> FitResult:
        """Fit a bifactor model by marginal maximum likelihood.

        Parameters
        ----------
        model : BifactorModel
            Model to fit in place. Coordinates fixed with
            ``set_free_parameter_masks`` keep their values.
        responses : ndarray of shape (n_persons, n_items)
            Dichotomous responses with negative values marking missing ones.
        start : {"default", "model"} or mapping, default="default"
            Starting values, as for :meth:`EMEstimator.fit`.

        Returns
        -------
        FitResult
            Fitted model, log-likelihood, standard errors and fit statistics.

        Raises
        ------
        MirtModelError
            If ``model`` is not a built-in ``BifactorModel``.
        MirtDataError
            If an observed response is not 0 or 1.
        """
        from mirt.estimation._refit import recipe_for
        from mirt.results.fit_result import FitResult

        _require_bifactor_model(model)
        start = _validate_start(start)
        responses = self._validate_responses(responses, model.n_items)
        if np.any(responses > 1):
            raise MirtDataError(
                "BifactorEMEstimator requires dichotomous 0/1 responses",
                n_persons=responses.shape[0],
                n_items=responses.shape[1],
            )
        _apply_starting_values(model, start)
        grid = _Grid.build(self.n_quadpts, model)

        with EMFitContext(responses, compress=True) as context:
            log_likelihood, converged, n_iterations = self._run_em(model, grid, context)
            model._is_fitted = True
            standard_errors: dict[str, NDArray[np.float64]] = {}
            se_method = covariance = None
            if self.compute_standard_errors:
                se_method = "oakes" if self.se_method == "auto" else self.se_method
                standard_errors, covariance = _standard_errors(
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
        model: BifactorModel,
        grid: _Grid,
        context: EMFitContext,
    ) -> tuple[float, bool, int]:
        """Iterate E- and M-steps and return the final log-likelihood and status."""
        self._convergence_history = []
        previous = -np.inf
        converged = False
        for iteration in range(self.max_iter):
            curves = _Curves.evaluate(grid, _coefficients(model))
            current, correct, observed = _expected_counts(grid, curves, context)
            self._convergence_history.append(current)
            self._log_iteration(iteration, current)
            if self._check_convergence(previous, current):
                converged = True
                break
            previous = current
            _m_step(model, grid, correct, observed)
        else:
            curves = _Curves.evaluate(grid, _coefficients(model))
            current, _, _ = _expected_counts(grid, curves, context)
            self._convergence_history.append(current)
            converged = self._check_convergence(previous, current)
        if self.verbose and converged:
            print(f"Converged at iteration {iteration}")
        return current, converged, iteration + 1

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_quadpts={self.n_quadpts}, "
            f"max_iter={self.max_iter}, tol={self.tol})"
        )


def _require_bifactor_model(model: object) -> None:
    from mirt.models.bifactor import BifactorModel

    if (
        type(model) is not BifactorModel
        or not uses_builtin_model_hooks(model, likelihood=True)
        or tuple(model._parameters) != _PARAMETERS
    ):
        raise MirtModelError(
            "BifactorEMEstimator requires a built-in BifactorModel; use "
            "EMEstimator for other or customized models",
            model_type=type(model).__name__,
        )


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
    lower, upper = bounds[:, 0], bounds[:, 1]
    estimable = np.any(observed > 0.0, axis=1) & np.any(free, axis=1)
    newton = np.flatnonzero(estimable & np.all(free, axis=1))
    bounded = np.flatnonzero(estimable & ~np.all(free, axis=1))
    updated = coefficients.copy()
    if newton.size:
        slopes, intercepts, solved = newton_logistic_items(
            grid.design[:, :2],
            correct[newton],
            observed[newton],
            coefficients[newton, :2],
            coefficients[newton, 2],
        )
        candidate = np.column_stack((slopes, intercepts))
        solved &= np.all((candidate >= lower) & (candidate <= upper), axis=1)
        updated[newton[solved]] = candidate[solved]
        bounded = np.concatenate((bounded, newton[~solved]))
    for item in bounded:
        updated[item] = _bounded_item_update(
            grid.design,
            correct[item],
            observed[item],
            coefficients[item],
            free[item],
            lower,
            upper,
        )
    model.set_parameters(
        **{name: updated[:, column] for column, name in enumerate(_PARAMETERS)}
    )


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
    """Accumulate the exact observed information from reduced posteriors.

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
    coefficients = _coefficients(model)
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
    from mirt.estimation.standard_errors import (
        _coordinates_at_bounds,
        _covariance,
        _flatten_parameters,
        _se_from_covariance,
        _unflatten_se,
    )

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


def bfactor(
    data: NDArray[np.int_] | Any,
    specific_factors: Sequence[int] | NDArray[np.int_],
    *,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    item_names: list[str] | None = None,
    compute_standard_errors: bool = True,
    se_method: StandardErrorMethod = "auto",
    start_values: Mapping[str, ArrayLike] | None = None,
    fixed: Mapping[str, ArrayLike] | None = None,
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
        Dichotomous responses. Negative values and ``NaN`` mark missing ones.
    specific_factors : sequence of int
        Specific-factor label of each item. Labels may be any non-negative
        integers; they are kept for reporting.
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
        or ``"intercepts"``. They also set the values of fixed coordinates.
    fixed : mapping of str to bool or array of bool, optional
        Coordinates to hold at their starting values. Fixing an item's
        specific loading at zero makes it load on the general factor only.

    Returns
    -------
    FitResult
        Fitted :class:`~mirt.models.BifactorModel` and fit statistics.

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
    """
    from mirt.estimation.base import _free_masks_from_fixed
    from mirt.models.bifactor import BifactorModel
    from mirt.utils.data import response_column_names, validate_responses

    if item_names is None:
        item_names = response_column_names(data)
    responses = validate_responses(data)
    n_items = responses.shape[1]
    if item_names is None:
        item_names = [f"Item_{index + 1}" for index in range(n_items)]
    try:
        model = BifactorModel(n_items, specific_factors, item_names=item_names)
    except MirtValidationError:
        raise
    except ValueError as exc:
        raise MirtValidationError(
            str(exc), parameter="specific_factors", value=specific_factors
        ) from exc
    if fixed is not None:
        model.set_free_parameter_masks(_free_masks_from_fixed(model, fixed))
    estimator = BifactorEMEstimator(
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        compute_standard_errors=compute_standard_errors,
        se_method=se_method,
    )
    return estimator.fit(
        model, responses, start="default" if start_values is None else start_values
    )
