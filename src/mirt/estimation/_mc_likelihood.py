"""Bounded likelihood reductions for person-specific Monte Carlo draws."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import qmc

from mirt._model_defaults import original_model_hook
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtValidationError
from mirt.models.base import BaseItemModel, DichotomousItemModel, PolytomousItemModel

_MAX_MC_LIKELIHOOD_ELEMENTS = 131_072
# Scrambled Sobol points (2**12) integrating the prior where EM has no
# default grid, beyond six factors.
_PRIOR_QMC_EXPONENT = 12


@dataclass(frozen=True)
class PriorGrid:
    """Nodes and weights that integrate against a Gaussian ability prior.

    Attributes
    ----------
    nodes : ndarray of shape (n_nodes, n_factors)
        Ability points.
    weights : ndarray of shape (n_nodes,)
        Positive prior mass of each node, summing to one.
    """

    nodes: NDArray[np.float64]
    weights: NDArray[np.float64]

    @property
    def n_total_points(self) -> int:
        """Number of nodes."""
        return len(self.weights)


def prior_grid(
    prior_mean: NDArray[np.float64],
    cholesky: NDArray[np.float64],
    rng: np.random.Generator,
) -> PriorGrid:
    """Return EM's default quadrature grid carried onto a Gaussian prior.

    Up to six factors, the standard-normal grid that ``EMEstimator`` uses by
    default (:func:`~mirt.estimation.quadrature.default_quadpts` points per
    dimension on the rule of
    :func:`~mirt.estimation.quadrature.build_quadrature`) is mapped through
    ``prior_mean + cholesky @ z``, so marginal log-likelihoods on it match
    EM's at the same parameters. More factors, which have no default grid,
    use 4,096 scrambled Sobol points with equal mass, scrambled from ``rng``.

    Parameters
    ----------
    prior_mean : ndarray of shape (n_factors,)
        Prior mean.
    cholesky : ndarray of shape (n_factors, n_factors)
        Lower Cholesky factor of the prior covariance.
    rng : numpy.random.Generator
        Scrambles the quasi-Monte Carlo points; unused up to six factors.

    Returns
    -------
    PriorGrid
        Read-only nodes and weights.
    """
    from mirt.estimation.quadrature import build_quadrature, default_quadpts

    n_factors = len(prior_mean)
    try:
        n_points = default_quadpts(n_factors)
    except MirtValidationError:
        sampler = qmc.Sobol(d=n_factors, scramble=True, seed=rng)
        uniforms = np.clip(
            sampler.random_base2(_PRIOR_QMC_EXPONENT),
            np.nextafter(0.0, 1.0),
            np.nextafter(1.0, 0.0),
        )
        standard = ndtri(uniforms)
        weights = np.full(len(standard), 1.0 / len(standard))
    else:
        quadrature = build_quadrature(n_points, n_factors)
        standard, weights = quadrature.nodes, quadrature.weights
    nodes = standard @ cholesky.T
    nodes += prior_mean
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return PriorGrid(nodes, weights)


_DEFAULT_BINARY_LIKELIHOOD = original_model_hook(DichotomousItemModel, "log_likelihood")
_DEFAULT_BINARY_VALIDATION = original_model_hook(
    DichotomousItemModel, "_validate_dichotomous_responses"
)
_DEFAULT_CATEGORY_LIKELIHOOD = original_model_hook(
    PolytomousItemModel, "log_likelihood"
)
_DEFAULT_THETA_VALIDATION = original_model_hook(BaseItemModel, "_ensure_theta_2d")
_DEFAULT_CATEGORY_VALIDATION = original_model_hook(
    PolytomousItemModel, "_validate_polytomous_responses"
)


def uses_default_sample_likelihood(model: BaseItemModel) -> bool:
    """Whether the sampled likelihood uses the ordinary model validation path."""
    if any(name in vars(model) for name in ("log_likelihood", "_ensure_theta_2d")):
        return False
    if type(model)._ensure_theta_2d is not _DEFAULT_THETA_VALIDATION:
        return False
    return (
        isinstance(model, DichotomousItemModel)
        and type(model).log_likelihood is _DEFAULT_BINARY_LIKELIHOOD
        and "_validate_dichotomous_responses" not in vars(model)
        and type(model)._validate_dichotomous_responses is _DEFAULT_BINARY_VALIDATION
    ) or (
        isinstance(model, PolytomousItemModel)
        and type(model).log_likelihood is _DEFAULT_CATEGORY_LIKELIHOOD
        and "_validate_polytomous_responses" not in vars(model)
        and type(model)._validate_polytomous_responses is _DEFAULT_CATEGORY_VALIDATION
    )


def sampled_log_likelihoods(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    samples: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Reduce ordinary item likelihoods without repeating the response matrix.

    Public probability curves retain their behavior. Custom likelihood or
    validation methods return ``None`` so their model-based evaluation remains
    authoritative. Only bounded point/response blocks are converted to floats.
    """
    if not uses_default_sample_likelihood(model):
        return None
    category_model = None
    if isinstance(model, DichotomousItemModel):
        responses = model._validate_dichotomous_responses(responses)
        width = model.n_items
    elif isinstance(model, PolytomousItemModel):
        category_model = model
        responses = model._validate_polytomous_responses(responses)
        width = max(model.n_categories)
    else:
        return None

    n_persons, n_samples, n_factors = samples.shape
    max_points = max(1, _MAX_MC_LIKELIHOOD_ELEMENTS // max(width, n_factors))
    sample_chunk = min(n_samples, max_points)
    row_chunk = max(
        1, min(max_points // sample_chunk, _MAX_MC_LIKELIHOOD_ELEMENTS // model.n_items)
    )

    def blocks() -> Iterator[tuple[slice, slice, NDArray[np.float64]]]:
        for first in range(0, n_persons, row_chunk):
            rows = slice(first, min(first + row_chunk, n_persons))
            for start in range(0, n_samples, sample_chunk):
                columns = slice(start, min(start + sample_chunk, n_samples))
                points = np.asarray(samples[rows, columns], dtype=np.float64).reshape(
                    -1, n_factors
                )
                yield rows, columns, points
                del points

    # Validate every draw before invoking probability callbacks, as the model
    # path does, without allocating a full converted sample tensor or mask.
    for _, _, points in blocks():
        if not np.all(np.isfinite(points)):
            raise ValueError("theta_samples must contain only finite values")
        del points

    result = np.empty((n_persons, n_samples), dtype=np.float64)
    for rows, columns, points in blocks():
        data = responses[rows]
        shape = (len(data), columns.stop - columns.start)
        if category_model is None:
            values = _binary_log_likelihoods(model, data, points, shape)
        else:
            values = _category_log_likelihoods(category_model, data, points, shape)
        if not np.all(np.isfinite(values)):
            raise ValueError("model.log_likelihood() returned invalid sampled values")
        result[rows, columns] = values
        del points, values
    return result


def _binary_log_likelihoods(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    points: NDArray[np.float64],
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Reduce bounded item curves against unexpanded responses."""
    observed = responses >= 0
    probabilities = np.clip(
        np.broadcast_to(model.probability(points), (len(points), model.n_items)),
        PROB_EPSILON,
        1.0 - PROB_EPSILON,
    )
    if probabilities.dtype == np.float64 and not np.any(
        observed & (responses != 0) & (responses != 1)
    ):
        # Ordinary binary responses need only their selected outcome. The
        # owned curve buffer becomes the item log likelihood before reduction.
        selected = probabilities.reshape(*shape, model.n_items)
        np.subtract(1.0, selected, out=selected, where=(responses == 0)[:, None, :])
        np.log(selected, out=selected)
        np.copyto(selected, 0.0, where=~observed[:, None, :])
        return selected.sum(axis=2)
    # Preserve the public curve's dtype and general response-value behavior.
    if responses.dtype.kind == "u":
        responses = responses.astype(
            np.result_type(responses.dtype, probabilities.dtype), copy=False
        )
    failure = np.log(1.0 - probabilities).reshape(*shape, model.n_items)
    np.log(probabilities, out=probabilities)
    success = probabilities.reshape(*shape, model.n_items)
    terms = success * responses[:, None, :]
    if responses.dtype.kind == "b":
        terms = terms.astype(np.result_type(np.int64, probabilities.dtype), copy=False)
    terms += failure * (1 - responses[:, None, :])
    np.copyto(terms, 0.0, where=~observed[:, None, :])
    return terms.sum(axis=2)


def _category_log_likelihoods(
    model: PolytomousItemModel,
    responses: NDArray[np.int_],
    points: NDArray[np.float64],
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Gather observed categories from bounded public item curves."""
    values = np.zeros(shape)
    row_indices = np.arange(shape[0])[:, None]
    sample_indices = np.arange(shape[1])[None, :]
    for item in range(model.n_items):
        decisions = responses[:, item]
        observed = np.flatnonzero(decisions >= 0)
        if not observed.size:
            continue
        probabilities = model.probability(points, item)
        probabilities = np.broadcast_to(
            probabilities, (len(points), probabilities.shape[1])
        ).reshape(*shape, probabilities.shape[1])
        categories = decisions[observed].astype(np.intp)
        selected = np.clip(
            probabilities[row_indices[observed], sample_indices, categories[:, None]],
            PROB_EPSILON,
            1.0,
        )
        np.log(selected, out=selected)
        values[observed] += selected
        del probabilities, selected
    return values
