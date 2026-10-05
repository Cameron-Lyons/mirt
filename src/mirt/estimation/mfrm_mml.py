"""Marginal maximum likelihood estimation of many-facet Rasch models."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import block_diag, solve_triangular
from scipy.optimize import minimize
from scipy.sparse import csr_array
from scipy.special import logsumexp

from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.mfrm import ManyFacetRaschModel, MFRMResult, PolytomousMFRM
from mirt.utils.data import validate_responses

# The person grid spans +-_GRID_LIMIT standard deviations.
_GRID_LIMIT = 6.0
# Automatic grids start at _AUTO_QUADPTS nodes and are refined, up to
# _MAX_QUADPTS, while the node spacing exceeds _SPACING_RATIO times the
# 5th percentile of the person posterior standard deviations.
_AUTO_QUADPTS = 41
_MAX_QUADPTS = 401
_SPACING_RATIO = 1.5
# log(sigma) is kept in a range where every grid evaluation stays finite.
_LOG_SIGMA_BOUNDS = (math.log(1e-4), math.log(1e3))


@dataclass(frozen=True)
class _Design:
    """Observed ratings grouped into cells sharing one item and facet levels."""

    n_persons: int
    n_categories: int
    person: NDArray[np.intp]
    cell: NDArray[np.intp]
    response: NDArray[np.intp]
    item: NDArray[np.intp]
    levels: tuple[NDArray[np.intp], ...]
    cell_item: NDArray[np.intp]
    cell_levels: tuple[NDArray[np.intp], ...]
    # Person-by-(cell, category) observation counts and their transpose.
    counts: csr_array
    counts_t: csr_array
    # Observations in each cell with a response of at least category j >= 1.
    cell_tail: NDArray[np.float64]


@dataclass(frozen=True)
class _ParameterMap:
    """Affine map from free coordinates to the natural parameter vector.

    The natural vector stacks item difficulties, every facet's levels, the
    threshold rows and log sigma. Anchored facets keep their mean at the
    anchor value and every threshold row sums to zero.
    """

    matrix: NDArray[np.float64]
    offset: NDArray[np.float64]
    items: slice
    facets: tuple[slice, ...]
    thresholds: slice
    threshold_rows: int

    def natural(self, free: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.matrix @ free + self.offset

    def free(self, natural: NDArray[np.float64]) -> NDArray[np.float64]:
        solution, *_ = np.linalg.lstsq(self.matrix, natural - self.offset, rcond=None)
        return solution

    def unpack(
        self, natural: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]], NDArray[np.float64]]:
        """Split a natural vector into items, facet levels and threshold rows."""
        thresholds = natural[self.thresholds].reshape(self.threshold_rows, -1)
        return (
            natural[self.items],
            [natural[block] for block in self.facets],
            thresholds,
        )


def fit_mfrm(
    model: ManyFacetRaschModel,
    responses: ArrayLike,
    facet_indices: Mapping[str, ArrayLike] | None = None,
    *,
    item_indices: ArrayLike | None = None,
    n_quadpts: int | None = None,
    estimate_sd: bool = True,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> MFRMResult:
    """Fit a many-facet Rasch model by marginal maximum likelihood.

    Person measures follow ``N(0, sigma^2)`` and are integrated out on an
    equally spaced grid. Item difficulties, facet levels, category thresholds
    and ``log(sigma)`` are estimated jointly by L-BFGS-B using the analytic
    (Fisher identity) gradient of the marginal log-likelihood.

    Parameters
    ----------
    model : ManyFacetRaschModel or PolytomousMFRM
        Model to estimate. Its parameters are replaced by the estimates and
        it is marked as fitted.
    responses : array-like of shape (n_persons, n_columns)
        Ratings coded ``0`` to ``n_categories - 1`` (``0``/``1`` for the
        binary model). Negative values and ``NaN`` mark missing ratings.
        Without ``item_indices`` the columns are the model's items.
    facet_indices : mapping, optional
        Level of every facet for each rating: a scalar, a per-person vector
        of shape ``(n_persons,)`` or an array of shape
        ``(n_persons, n_columns)``. Entries at missing ratings are ignored.
    item_indices : array-like, optional
        Item rated in each column, of shape ``(n_columns,)`` or
        ``(n_persons, n_columns)``. This long layout lets several raters
        score the same person and item; entries at missing ratings are
        ignored.
    n_quadpts : int, optional
        Number of grid nodes for the person measures (at least 5). By default
        the grid starts at 41 nodes and is refined, up to 401, until the node
        spacing resolves the person posteriors; an explicit value is used as
        given.
    estimate_sd : bool, default=True
        Estimate the person standard deviation ``sigma``; otherwise it is
        fixed at 1.
    max_iter : int, default=500
        Maximum number of L-BFGS-B iterations, summed over grid refinements.
    tol : float, default=1e-6
        Convergence tolerance on the largest gradient component of the
        per-person negative log-likelihood.

    Returns
    -------
    MFRMResult
        Estimates with standard errors from the inverse observed information,
        infit and outfit mean squares for every item and facet level, EAP
        person measures with posterior standard deviations, information
        criteria and the number of grid nodes used.

    Raises
    ------
    MirtValidationError
        If a facet is not anchored, the arguments are invalid, or items and
        facet levels are not jointly identified by the rating design (for
        example a facet confounded with items).
    MirtDataError
        If the ratings are invalid, an item or facet level has no ratings or
        only extreme ratings, a response category is never used, or no person
        has two ratings while ``sigma`` is estimated.

    Warns
    -----
    RuntimeWarning
        If the final grid is too coarse for the person posteriors, so that
        the estimates may carry discretization bias.

    Notes
    -----
    The rating of person ``n`` on item ``i`` with facet levels ``l`` has

    .. math::

        \\log \\frac{P(X = k)}{P(X = k - 1)}
            = \\theta_n - b_i - \\sum_f d_{f l_f} - \\tau_{(i) k},

    with ``tau`` shared across items (rating scale) or item specific
    (partial credit). Each threshold row sums to zero and each facet's
    levels average to its ``anchor_value``. The binary model has no
    thresholds.

    The person grid spans ``+-6 sigma`` with normal weights (the trapezoidal
    rule). Its error falls off like ``exp(-2 pi^2 (s / h)^2)`` for posterior
    standard deviation ``s`` and node spacing ``h``, so a grid that resolves
    the posteriors is far more accurate than Gauss-Hermite nodes when many
    ratings per person make the posteriors narrow.

    Standard errors come from the exact observed information of the grid
    log-likelihood (Louis' identity) and include the levels fixed by the
    anchoring constraint.

    Infit and outfit average the squared (standardized) residuals over each
    person's posterior, the quadrature counterpart of plausible-value fit
    statistics, so both have expectation one under the model. Residuals at
    point estimates of the person measures would bias them downward when
    persons have few ratings.

    References
    ----------
    Linacre, J. M. (1994). *Many-Facet Rasch Measurement*. MESA Press.

    Examples
    --------
    >>> import numpy as np
    >>> from mirt.estimation import fit_mfrm
    >>> from mirt.models import Facet, PolytomousMFRM
    >>> rng = np.random.default_rng(0)
    >>> truth = PolytomousMFRM(6, 4, [Facet("rater", 4)])
    >>> _ = truth.set_facet_parameters("rater", np.array([-0.5, 0.0, 0.2, 0.3]))
    >>> raters = rng.integers(0, 4, size=(300, 6))
    >>> responses = truth.simulate(rng.normal(size=300), {"rater": raters}, seed=1)
    >>> model = PolytomousMFRM(6, 4, [Facet("rater", 4)])
    >>> result = fit_mfrm(model, responses, {"rater": raters})
    >>> result.facet_parameters["rater"].shape
    (4,)
    """
    if not isinstance(model, ManyFacetRaschModel):
        raise MirtValidationError(
            "model must be a ManyFacetRaschModel or PolytomousMFRM",
            parameter="model",
        )
    for facet in model.facets:
        if not facet.is_anchored:
            raise MirtValidationError(
                f"facet '{facet.name}' is not anchored; with free item "
                "difficulties and a person mean of zero its levels are not "
                "identified, so set is_anchored=True",
                parameter="facets",
            )
    if n_quadpts is not None:
        n_quadpts = _validated_integer(n_quadpts, "n_quadpts", minimum=5)
    max_iter = _validated_integer(max_iter, "max_iter", minimum=1)
    if not isinstance(estimate_sd, (bool, np.bool_)):
        raise MirtValidationError(
            "estimate_sd must be a boolean", parameter="estimate_sd"
        )
    if (
        isinstance(tol, (bool, np.bool_))
        or not isinstance(tol, (int, float, np.integer, np.floating))
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise MirtValidationError(
            "tol must be a positive finite number", parameter="tol", value=tol
        )

    design = _build_design(model, responses, facet_indices, item_indices)
    _check_estimability(model, design, bool(estimate_sd))
    parameter_map = _parameter_map(model, design.n_categories, bool(estimate_sd))
    _check_identification(model, design, parameter_map)

    free = parameter_map.free(_starting_values(model, design, parameter_map))
    bounds = [(None, None)] * free.size
    if estimate_sd:
        bounds[-1] = _LOG_SIGMA_BOUNDS
    n_nodes = _AUTO_QUADPTS if n_quadpts is None else n_quadpts
    n_iterations = 0
    while True:
        likelihood = _MarginalLikelihood(design, parameter_map, n_nodes)
        optimum = minimize(
            likelihood.objective,
            free,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": max_iter - n_iterations,
                "gtol": float(tol),
                "ftol": 1e-14,
            },
        )
        free = np.asarray(optimum.x, dtype=np.float64)
        n_iterations += int(optimum.nit)
        log_likelihood, _, posterior = likelihood.evaluate(free, gradient=False)
        needed = likelihood.required_nodes(free, posterior)
        if (
            n_quadpts is not None
            or needed <= n_nodes
            or n_nodes == _MAX_QUADPTS
            or n_iterations >= max_iter
        ):
            break
        # Posteriors on a coarse grid look narrower than they are, so the
        # grid grows at most threefold per refinement.
        n_nodes = min(needed, 3 * n_nodes - 2, _MAX_QUADPTS)
    if needed > n_nodes:
        warnings.warn(
            f"the {n_nodes}-node person grid is coarser than the person "
            "posteriors, so the estimates may be biased; increase n_quadpts",
            RuntimeWarning,
            stacklevel=2,
        )

    covariance = _natural_covariance(likelihood.information(free), parameter_map)
    natural = parameter_map.natural(free)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return _result(
        model,
        design,
        parameter_map,
        likelihood,
        natural,
        standard_errors,
        posterior,
        log_likelihood=log_likelihood,
        n_parameters=free.size,
        n_iterations=n_iterations,
        converged=bool(optimum.success),
    )


def _validated_integer(value: object, parameter: str, *, minimum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise MirtValidationError(
            f"{parameter} must be an integer of at least {minimum}",
            parameter=parameter,
            value=value,
        )
    return int(value)


def _observed_index(
    values: ArrayLike,
    name: str,
    n_levels: int,
    observed: NDArray[np.bool_],
    *,
    vector_axis: int,
) -> NDArray[np.intp]:
    """Broadcast an index array to the response grid and keep observed entries."""
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise MirtValidationError(f"{name} must contain integers", parameter=name)
    shape = observed.shape
    if array.ndim == 1 and array.shape[0] == shape[vector_axis]:
        array = array[:, None] if vector_axis == 0 else array[None, :]
    elif (
        array.ndim == 1 or array.ndim > 2 or (array.ndim == 2 and array.shape != shape)
    ):
        expected = f"({shape[vector_axis]},) or {shape}"
        raise MirtValidationError(
            f"{name} must be a scalar or have shape {expected}; got {array.shape}",
            parameter=name,
        )
    selected = np.broadcast_to(array, shape)[observed]
    if np.any((selected < 0) | (selected >= n_levels)):
        raise MirtValidationError(
            f"{name} must lie in [0, {n_levels}) at observed ratings",
            parameter=name,
        )
    return selected.astype(np.intp)


def _build_design(
    model: ManyFacetRaschModel,
    responses: ArrayLike,
    facet_indices: Mapping[str, ArrayLike] | None,
    item_indices: ArrayLike | None,
) -> _Design:
    """Flatten observed ratings and group them by item and facet levels."""
    data = validate_responses(
        responses, n_items=model.n_items if item_indices is None else None
    )
    n_persons, n_columns = data.shape
    observed = data >= 0
    if not np.any(observed):
        raise MirtDataError("responses contain no observed ratings")
    n_categories = model.n_categories if isinstance(model, PolytomousMFRM) else 2
    response = data[observed].astype(np.intp)
    if np.any(response >= n_categories):
        raise MirtDataError(
            f"ratings must be coded 0 to {n_categories - 1}",
            n_categories=n_categories,
        )

    person = np.broadcast_to(np.arange(n_persons)[:, None], data.shape)[observed]
    if item_indices is None:
        item = np.broadcast_to(np.arange(n_columns), data.shape)[observed]
    else:
        item = _observed_index(
            item_indices, "item_indices", model.n_items, observed, vector_axis=1
        )

    provided = set(facet_indices or {})
    unknown = sorted(provided - set(model.facet_names))
    missing = sorted(set(model.facet_names) - provided)
    if unknown:
        raise MirtValidationError(f"Unknown facet assignments: {', '.join(unknown)}")
    if missing:
        raise MirtValidationError(f"Missing facet assignments: {', '.join(missing)}")
    levels = tuple(
        _observed_index(
            facet_indices[facet.name],
            f"facet '{facet.name}' indices",
            facet.n_levels,
            observed,
            vector_axis=0,
        )
        for facet in model.facets
        if facet_indices is not None
    )

    keys = np.column_stack((item, *levels))
    unique_keys, cell = np.unique(keys, axis=0, return_inverse=True)
    cell = cell.reshape(-1).astype(np.intp)
    n_cells = len(unique_keys)
    flat = cell * n_categories + response
    counts = csr_array(
        (np.ones(len(flat)), (person, flat)),
        shape=(n_persons, n_cells * n_categories),
    )
    histogram = np.bincount(flat, minlength=n_cells * n_categories).reshape(
        n_cells, n_categories
    )
    cell_tail = np.cumsum(histogram[:, :0:-1], axis=1)[:, ::-1].astype(np.float64)
    return _Design(
        n_persons=n_persons,
        n_categories=n_categories,
        person=person.astype(np.intp),
        cell=cell,
        response=response,
        item=item.astype(np.intp),
        levels=levels,
        cell_item=unique_keys[:, 0].astype(np.intp),
        cell_levels=tuple(
            unique_keys[:, index + 1].astype(np.intp) for index in range(len(levels))
        ),
        counts=counts,
        counts_t=csr_array(counts.T),
        cell_tail=cell_tail,
    )


def _check_estimability(
    model: ManyFacetRaschModel, design: _Design, estimate_sd: bool
) -> None:
    """Reject items, levels and categories whose estimates would diverge."""
    top = design.n_categories - 1
    groups = [("item", model.item_names, design.item)] + [
        (f"facet '{facet.name}' level", list(facet.labels or []), levels)
        for facet, levels in zip(model.facets, design.levels, strict=True)
    ]
    for kind, labels, index in groups:
        count = np.bincount(index, minlength=len(labels))
        score = np.bincount(index, weights=design.response, minlength=len(labels))
        for position, label in enumerate(labels):
            if count[position] == 0:
                raise MirtDataError(f"{kind} '{label}' has no observed ratings")
            if score[position] in (0, top * count[position]):
                raise MirtDataError(
                    f"{kind} '{label}' has only extreme ratings, so its "
                    "measure is not estimable"
                )

    if isinstance(model, PolytomousMFRM):
        if model.category_structure == "partial_credit":
            table = np.zeros((model.n_items, design.n_categories), dtype=np.int64)
            np.add.at(table, (design.item, design.response), 1)
            empty = np.argwhere(table == 0)
            if len(empty):
                item, category = empty[0]
                raise MirtDataError(
                    f"category {category} is never used for item "
                    f"'{model.item_names[item]}'; collapse categories or use "
                    "category_structure='rating_scale'"
                )
        else:
            used = np.bincount(design.response, minlength=design.n_categories)
            if np.any(used == 0):
                raise MirtDataError(
                    f"category {int(np.argmin(used))} is never used; collapse "
                    "categories or reduce n_categories"
                )

    if estimate_sd and np.max(np.bincount(design.person)) < 2:
        raise MirtDataError(
            "sigma is not identified because no person has two ratings; "
            "use estimate_sd=False"
        )


def _parameter_map(
    model: ManyFacetRaschModel, n_categories: int, estimate_sd: bool
) -> _ParameterMap:
    """Build the constrained parameterization of items, facets and thresholds."""

    def centered(size: int, mean: float) -> tuple[NDArray, NDArray]:
        matrix = np.vstack((np.eye(size - 1), -np.ones((1, size - 1))))
        offset = np.zeros(size)
        offset[-1] = size * mean
        return matrix, offset

    blocks = [(np.eye(model.n_items), np.zeros(model.n_items))]
    blocks += [centered(facet.n_levels, facet.anchor_value) for facet in model.facets]
    threshold_rows = (
        model.n_items
        if isinstance(model, PolytomousMFRM)
        and model.category_structure == "partial_credit"
        else 1
    )
    blocks += [centered(n_categories - 1, 0.0) for _ in range(threshold_rows)]
    blocks.append((np.ones((1, int(estimate_sd))), np.zeros(1)))

    sizes = np.cumsum([0] + [len(offset) for _, offset in blocks])
    n_facets = model.n_facets
    return _ParameterMap(
        matrix=block_diag(*(matrix for matrix, _ in blocks)),
        offset=np.concatenate([offset for _, offset in blocks]),
        items=slice(0, model.n_items),
        facets=tuple(
            slice(sizes[index + 1], sizes[index + 2]) for index in range(n_facets)
        ),
        thresholds=slice(sizes[n_facets + 1], sizes[-2]),
        threshold_rows=threshold_rows,
    )


def _check_identification(
    model: ManyFacetRaschModel, design: _Design, parameter_map: _ParameterMap
) -> None:
    """Require the cell locations to determine every free item and facet value."""
    n_location = parameter_map.thresholds.start
    location_map = parameter_map.matrix[:n_location]
    location_map = location_map[:, np.any(location_map != 0, axis=0)]
    columns = np.column_stack(
        (
            design.cell_item,
            *(
                block.start + levels
                for block, levels in zip(
                    parameter_map.facets, design.cell_levels, strict=True
                )
            ),
        )
    )
    n_cells, n_terms = columns.shape
    indicator = csr_array(
        (
            np.ones(columns.size),
            (np.repeat(np.arange(n_cells), n_terms), columns.ravel()),
        ),
        shape=(n_cells, n_location),
    )
    gram = location_map.T @ (indicator.T @ indicator @ location_map)
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues[0] <= 1e-9 * max(eigenvalues[-1], 1.0):
        raise MirtValidationError(
            "item and facet parameters are not identified by this rating "
            "design; a facet is confounded with items or other facets "
            f"({', '.join(model.facet_names)})",
            parameter="facet_indices",
        )


def _starting_values(
    model: ManyFacetRaschModel, design: _Design, parameter_map: _ParameterMap
) -> NDArray[np.float64]:
    """Return logit-scaled marginal difficulties and adjacent-category thresholds."""
    top = design.n_categories - 1

    def difficulty(index: NDArray[np.intp], size: int) -> NDArray[np.float64]:
        count = np.bincount(index, minlength=size)
        proportion = np.bincount(index, weights=design.response, minlength=size) / (
            top * count
        )
        return np.log1p(-proportion) - np.log(proportion)

    natural = parameter_map.offset.copy()
    anchors = sum(facet.anchor_value for facet in model.facets)
    natural[parameter_map.items] = difficulty(design.item, model.n_items) - anchors
    for facet, block, levels in zip(
        model.facets, parameter_map.facets, design.levels, strict=True
    ):
        values = difficulty(levels, facet.n_levels)
        natural[block] = values - values.mean() + facet.anchor_value

    rows = parameter_map.threshold_rows
    table = np.ones((rows, design.n_categories))
    np.add.at(table, (design.item if rows > 1 else 0, design.response), 1.0)
    steps = np.log(table[:, :-1]) - np.log(table[:, 1:])
    natural[parameter_map.thresholds] = (
        steps - steps.mean(axis=1, keepdims=True)
    ).ravel()
    return natural


class _MarginalLikelihood:
    """Marginal log-likelihood and its Fisher-identity gradient."""

    def __init__(
        self, design: _Design, parameter_map: _ParameterMap, n_quadpts: int
    ) -> None:
        nodes = np.linspace(-_GRID_LIMIT, _GRID_LIMIT, n_quadpts)
        self.design = design
        self.parameter_map = parameter_map
        self.nodes = nodes
        self.log_weights = -0.5 * nodes**2 - logsumexp(-0.5 * nodes**2)
        self.categories = np.arange(design.n_categories, dtype=np.float64)

    def objective(self, free: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Return the per-person negative log-likelihood and its gradient."""
        log_likelihood, gradient, _ = self.evaluate(free)
        n_persons = self.design.n_persons
        return -log_likelihood / n_persons, -gradient / n_persons

    def person_moments(
        self, sigma: float, posterior: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the posterior means and standard deviations of the persons."""
        nodes = sigma * self.nodes
        mean = posterior @ nodes
        return mean, np.sqrt(np.maximum(posterior @ nodes**2 - mean**2, 0.0))

    def required_nodes(
        self, free: NDArray[np.float64], posterior: NDArray[np.float64]
    ) -> int:
        """Return the grid size needed to resolve the person posteriors.

        This is the current size while the node spacing is at most
        ``_SPACING_RATIO`` times the 5th percentile of the posterior standard
        deviations, and otherwise the size whose spacing equals it.
        """
        sigma = float(np.exp(self.parameter_map.natural(free)[-1]))
        _, deviation = self.person_moments(sigma, posterior)
        resolution = float(np.quantile(deviation, 0.05))
        span = sigma * (self.nodes[-1] - self.nodes[0])
        if span <= _SPACING_RATIO * resolution * (len(self.nodes) - 1):
            return len(self.nodes)
        return int(np.ceil(min(span / max(resolution, 1e-12), 1e9))) + 1

    def probabilities(
        self, natural: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Return log and plain category probabilities and sigma.

        The probability arrays have shape ``(n_cells, n_categories, n_nodes)``.
        """
        design = self.design
        items, facets, thresholds = self.parameter_map.unpack(natural)
        sigma = float(np.exp(natural[-1]))
        location = items[design.cell_item]
        for values, levels in zip(facets, design.cell_levels, strict=True):
            location = location + values[levels]
        cumulative = np.concatenate(
            (np.zeros((len(thresholds), 1)), np.cumsum(thresholds, axis=1)), axis=1
        )
        if len(cumulative) > 1:
            cumulative = cumulative[design.cell_item]
        eta = sigma * self.nodes - location[:, None]
        logits = self.categories[:, None] * eta[:, None, :] - cumulative[:, :, None]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        total = np.sum(exponentials, axis=1, keepdims=True)
        return shifted - np.log(total), exponentials / total, sigma

    def posterior(
        self, log_probability: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Return the marginal log-likelihood and the person posteriors."""
        table = log_probability.reshape(-1, log_probability.shape[2])
        joint = self.design.counts @ table + self.log_weights
        peak = np.max(joint, axis=1, keepdims=True)
        weights = np.exp(joint - peak)
        total = np.sum(weights, axis=1, keepdims=True)
        return float(np.sum(peak + np.log(total))), weights / total

    def expected_counts(self, posterior: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return posterior-expected ratings per cell, category and node."""
        n_categories = self.design.n_categories
        expected = self.design.counts_t @ posterior
        return expected.reshape(-1, n_categories, posterior.shape[1])

    def evaluate(
        self, free: NDArray[np.float64], *, gradient: bool = True
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """Return the log-likelihood, its free gradient and person posteriors."""
        design = self.design
        parameter_map = self.parameter_map
        natural = parameter_map.natural(free)
        log_probability, probability, sigma = self.probabilities(natural)
        log_likelihood, posterior = self.posterior(log_probability)
        if not gradient:
            return log_likelihood, np.empty(0), posterior

        expected = self.expected_counts(posterior)
        cell_node_count = np.sum(expected, axis=1)
        cell_node_score = np.einsum("k,ckq->cq", self.categories, expected)
        mean = np.einsum("k,ckq->cq", self.categories, probability)
        residual = cell_node_score - cell_node_count * mean
        location_gradient = -np.sum(residual, axis=1)

        natural_gradient = np.zeros_like(natural)
        natural_gradient[parameter_map.items] = np.bincount(
            design.cell_item,
            weights=location_gradient,
            minlength=parameter_map.items.stop,
        )
        for block, levels in zip(parameter_map.facets, design.cell_levels, strict=True):
            natural_gradient[block] = np.bincount(
                levels, weights=location_gradient, minlength=block.stop - block.start
            )
        # P(X >= j) for j = 1, ..., n_categories - 1.
        tail = np.cumsum(probability[:, :0:-1], axis=1)[:, ::-1]
        threshold_gradient = (
            np.einsum("cq,cjq->cj", cell_node_count, tail) - design.cell_tail
        )
        if parameter_map.threshold_rows > 1:
            rows = np.zeros((parameter_map.threshold_rows, design.n_categories - 1))
            np.add.at(rows, design.cell_item, threshold_gradient)
        else:
            rows = threshold_gradient.sum(axis=0)
        natural_gradient[parameter_map.thresholds] = rows.ravel()
        natural_gradient[-1] = sigma * np.sum(residual @ self.nodes)
        return log_likelihood, parameter_map.matrix.T @ natural_gradient, posterior

    def information(self, free: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the observed information of the free coordinates (Louis).

        Each rating is an exponential family in its local coordinates
        ``(eta, tau_1, ..., tau_J)`` with statistic
        ``g(k) = (k, -1[k >= 1], ..., -1[k >= J])`` and
        ``eta = sigma * z - location``. The information is the posterior
        expected complete-data information minus the posterior covariance of
        each person's complete-data score.
        """
        design = self.design
        parameter_map = self.parameter_map
        natural = parameter_map.natural(free)
        log_probability, probability, sigma = self.probabilities(natural)
        _, posterior = self.posterior(log_probability)
        expected = self.expected_counts(posterior)
        n_cells, n_categories, _ = probability.shape
        n_natural = len(natural)
        nodes = self.nodes

        statistic = np.tril(-np.ones((n_categories, n_categories)))
        statistic[:, 0] = self.categories
        mean = np.einsum("ckq,kl->clq", probability, statistic)
        count = np.sum(expected, axis=1)
        residual = np.einsum("k,ckq->cq", self.categories, expected) - (
            count * mean[:, 0]
        )
        # Count-weighted sums over nodes of the local covariance Cov(g(K)),
        # and of its eta column times z and its eta variance times z^2.
        weighted = np.einsum("cq,ckq->ck", count, probability)
        covariance = np.einsum(
            "ck,kl,km->clm", weighted, statistic, statistic
        ) - np.matmul(count[:, None, :] * mean, mean.transpose(0, 2, 1))
        weighted = np.einsum("cq,ckq->ck", count * nodes, probability)
        eta_covariance = weighted @ (statistic * statistic[:, :1]) - np.einsum(
            "cq,clq->cl", count * nodes * mean[:, 0], mean
        )
        eta_variance = np.einsum(
            "cq,ckq,k->c", count * nodes**2, probability, self.categories**2
        ) - np.einsum("cq,cq->c", count * nodes**2, mean[:, 0] ** 2)

        # Natural index, local coordinate and sign of each cell's parameters:
        # the item and facet levels enter eta negatively, thresholds as is.
        thresholds_row = (
            design.cell_item
            if parameter_map.threshold_rows > 1
            else np.zeros(n_cells, dtype=np.intp)
        )
        n_steps = n_categories - 1
        index = np.column_stack(
            (
                design.cell_item,
                *(
                    block.start + levels
                    for block, levels in zip(
                        parameter_map.facets, design.cell_levels, strict=True
                    )
                ),
                parameter_map.thresholds.start
                + n_steps * thresholds_row[:, None]
                + np.arange(n_steps),
            )
        )
        n_locations = 1 + len(design.cell_levels)
        local = np.r_[np.zeros(n_locations, dtype=np.intp), np.arange(1, n_categories)]
        sign = np.r_[-np.ones(n_locations), np.ones(n_steps)]

        pairs = (index[:, :, None] * n_natural + index[:, None, :]).ravel()
        blocks = sign[:, None] * sign * covariance[:, local][:, :, local]
        information = np.bincount(
            pairs, weights=blocks.ravel(), minlength=n_natural**2
        ).reshape(n_natural, n_natural)
        cross = sigma * np.bincount(
            index.ravel(),
            weights=(sign * eta_covariance[:, local]).ravel(),
            minlength=n_natural,
        )
        information[:, -1] += cross
        information[-1, :] += cross
        information[-1, -1] += sigma**2 * np.sum(eta_variance)
        # Second derivative of eta = sigma * z - location in log(sigma).
        information[-1, -1] -= sigma * np.sum(residual @ nodes)

        # Posterior covariance of the complete-data scores. Their node-varying
        # part is minus the local means E[g | z] of each rating mapped to its
        # cell's parameters, plus sigma * z * (x - E[x]) in log(sigma).
        person_cells = csr_array(
            (np.ones(len(design.cell)), (design.person, design.cell)),
            shape=(design.n_persons, n_cells),
        )
        total = np.bincount(
            design.person, weights=design.response, minlength=design.n_persons
        )
        pointers = np.arange(0, index.size + 1, index.shape[1])
        score_mean = np.zeros((design.n_persons, n_natural))
        for node in range(len(nodes)):
            loadings = csr_array(
                ((sign * mean[:, local, node]).ravel(), index.ravel(), pointers),
                shape=(n_cells, n_natural),
            )
            score = -(person_cells @ loadings).toarray()
            score[:, -1] = (
                sigma * nodes[node] * (total - person_cells @ mean[:, 0, node])
            )
            weighted_score = posterior[:, node, None] * score
            information -= score.T @ weighted_score
            score_mean += weighted_score
        information += score_mean.T @ score_mean
        return parameter_map.matrix.T @ information @ parameter_map.matrix


def _natural_covariance(
    information: NDArray[np.float64], parameter_map: _ParameterMap
) -> NDArray[np.float64]:
    """Map the inverse free-coordinate information to the natural parameters.

    Returns NaN entries when the information is not positive definite.
    """
    n_natural = len(parameter_map.offset)
    try:
        factor = np.linalg.cholesky(0.5 * (information + information.T))
    except np.linalg.LinAlgError:
        return np.full((n_natural, n_natural), np.nan)
    root = solve_triangular(factor, parameter_map.matrix.T, lower=True)
    return root.T @ root


def _result(
    model: ManyFacetRaschModel,
    design: _Design,
    parameter_map: _ParameterMap,
    likelihood: _MarginalLikelihood,
    natural: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
    posterior: NDArray[np.float64],
    *,
    log_likelihood: float,
    n_parameters: int,
    n_iterations: int,
    converged: bool,
) -> MFRMResult:
    """Store the estimates on the model and compute residual fit statistics."""
    items, facets, thresholds = parameter_map.unpack(natural)
    item_errors, facet_errors, threshold_errors = parameter_map.unpack(standard_errors)

    # Squared residuals averaged over each person's posterior, per cell.
    _, probability, sigma = likelihood.probabilities(natural)
    theta, theta_se = likelihood.person_moments(sigma, posterior)
    categories = likelihood.categories
    mean = np.einsum("k,ckq->cq", categories, probability)
    squared = (categories[:, None] - mean[:, None, :]) ** 2
    variance = np.maximum(
        np.sum(probability * squared, axis=1), np.finfo(np.float64).tiny
    )
    expected = likelihood.expected_counts(posterior)
    cell_squared = np.einsum("ckq,ckq->c", expected, squared)
    cell_standardized = np.einsum(
        "ckq,ckq->c", expected, squared / variance[:, None, :]
    )
    cell_variance = np.einsum("ckq,cq->c", expected, variance)
    cell_count = np.bincount(design.cell, minlength=len(design.cell_item))

    def mean_squares(
        index: NDArray[np.intp], size: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        infit = np.bincount(index, cell_squared, size) / np.bincount(
            index, cell_variance, size
        )
        outfit = np.bincount(index, cell_standardized, size) / np.bincount(
            index, cell_count, size
        )
        return infit, outfit

    model.set_item_difficulty(items)
    infit: dict[str, NDArray[np.float64]] = {}
    outfit: dict[str, NDArray[np.float64]] = {}
    for facet, values, levels in zip(
        model.facets, facets, design.cell_levels, strict=True
    ):
        model.set_facet_parameters(facet.name, values)
        infit[facet.name], outfit[facet.name] = mean_squares(levels, facet.n_levels)
    item_infit, item_outfit = mean_squares(design.cell_item, model.n_items)

    estimated_thresholds = None
    estimated_threshold_errors = None
    if isinstance(model, PolytomousMFRM):
        shape = model.thresholds.shape
        estimated_thresholds = thresholds.reshape(shape)
        estimated_threshold_errors = threshold_errors.reshape(shape)
        model.set_thresholds(estimated_thresholds)
    model._is_fitted = True

    if not parameter_map.matrix[-1].any():
        sigma_error = 0.0
    elif np.any(np.isclose(natural[-1], _LOG_SIGMA_BOUNDS, rtol=0.0, atol=1e-8)):
        # A boundary estimate (persons that do not differ) has no Wald error.
        sigma_error = float("nan")
    else:
        sigma_error = sigma * float(standard_errors[-1])
    return MFRMResult(
        model=model,
        facet_parameters=model.facet_parameters,
        facet_se={
            facet.name: errors
            for facet, errors in zip(model.facets, facet_errors, strict=True)
        },
        infit=infit,
        outfit=outfit,
        log_likelihood=log_likelihood,
        n_iterations=n_iterations,
        converged=converged,
        item_difficulty=items.copy(),
        item_se=item_errors.copy(),
        item_infit=item_infit,
        item_outfit=item_outfit,
        thresholds=estimated_thresholds,
        threshold_se=estimated_threshold_errors,
        sigma=sigma,
        sigma_se=sigma_error,
        theta=theta,
        theta_se=theta_se,
        n_parameters=n_parameters,
        n_observations=len(design.response),
        n_quadpts=len(likelihood.nodes),
    )
