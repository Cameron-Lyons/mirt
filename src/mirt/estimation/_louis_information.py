"""Exact marginal information for item models from item-local derivatives.

Louis (1982) writes the observed information of a mixture likelihood as the
posterior expectation of the complete-data information minus the posterior
variance of the complete-data score. On a quadrature grid with a fixed prior
mass both terms are finite sums, so the identity is exact. Only derivatives
of each item's category log-probabilities at the nodes are needed. Built-in
unidimensional logistic, graded, partial-credit and rating-scale items use
closed forms. Other built-in item models difference one item's curve at a
time, which costs a few dozen curve evaluations per item rather than O(P^2)
marginal likelihood evaluations. Parameters shared by all items, such as
rating-scale thresholds, enter every item's derivatives and are accumulated
once. Latent-density parameters are treated as fixed.

References
----------
Louis, T. A. (1982). Finding the observed information matrix when using the
    EM algorithm. Journal of the Royal Statistical Society B, 44, 226-233.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt._model_defaults import uses_builtin_model_hooks
from mirt.constants import PROB_EPSILON
from mirt.estimation._posterior import normalize_log_posterior

if TYPE_CHECKING:
    from mirt.estimation.standard_errors import _ParameterLayout
    from mirt.models.base import BaseItemModel
    from mirt.models.mixed_format import MixedItemModel

# Person-by-node-by-parameter scratch for one block of score vectors.
_MAX_BLOCK_ENTRIES = 1 << 22
# Relative step for item-local central differences of log-probabilities.
_DERIVATIVE_STEP = 1e-4
# Expected (Fisher) information enumerates every response pattern.
MAX_ENUMERATED_PATTERNS = 1 << 16


@dataclass(frozen=True)
class _ItemTerms:
    """Derivatives of one item's category log-probabilities at the nodes.

    ``first`` has shape ``(k, Q, C)`` and ``second`` ``(k, k, Q, C)`` for the
    item's ``k`` free coordinates, which occupy ``columns`` of the free
    parameter vector. The last ``n_shared`` coordinates are shared by every
    item and appear in the same order in each item's terms.
    """

    columns: NDArray[np.intp]
    first: NDArray[np.float64]
    second: NDArray[np.float64]
    n_shared: int = 0


@dataclass(frozen=True)
class LouisInformation:
    """Information terms accumulated in one pass over the persons.

    Attributes
    ----------
    information : ndarray of shape (P, P)
        Weighted sum of person observed information matrices.
    score_crossproduct : ndarray of shape (P, P)
        ``sum_i m_i s_i s_i'`` of the exact marginal person scores, where
        ``m`` is ``meat_weights`` or, by default, the person weights.
    """

    information: NDArray[np.float64]
    score_crossproduct: NDArray[np.float64]


def _analytic_model_types() -> tuple[type, ...]:
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        OneParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        GradedResponseModel,
        PartialCreditModel,
    )

    return (
        OneParameterLogistic,
        TwoParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
        GradedResponseModel,
        GeneralizedPartialCredit,
        PartialCreditModel,
    )


def has_analytic_item_derivatives(model: BaseItemModel) -> bool:
    """Whether closed-form item derivatives describe ``model`` exactly."""
    from mirt.models.polytomous import _uses_authored_rating_scale_hooks

    if model.n_factors != 1:
        return False
    if type(model) in _analytic_model_types():
        return uses_builtin_model_hooks(model, likelihood=True)
    return _uses_authored_rating_scale_hooks(model)


def supports_louis_information(model: BaseItemModel) -> bool:
    """Whether each free parameter belongs to one item or to all items.

    Exact built-in types keep the person likelihood a product of item curves
    evaluated by ``probability``, which item-local derivatives require. Of
    those, only the rating-scale families have parameters shared by all
    items. A mixed-format model qualifies when each of its components does
    and none has shared parameters.
    """
    from mirt.estimation._patterns import supports_pattern_compression
    from mirt.models.mixed_format import MixedItemModel, uses_component_likelihoods
    from mirt.models.polytomous import _uses_authored_rating_scale_hooks

    if isinstance(model, MixedItemModel):
        return (
            uses_component_likelihoods(model)
            and all(map(supports_louis_information, model.component_models))
            and not any(
                getattr(component, "_shared_parameters", frozenset())
                for component in model.component_models
            )
        )
    if _uses_authored_rating_scale_hooks(model):
        return True
    if not supports_pattern_compression(model):
        return False
    return all(model._item_indexed(name) for name in model._parameters)


def _item_categories(model: BaseItemModel) -> list[int]:
    if not model.is_polytomous:
        return [2] * model.n_items
    counts = np.broadcast_to(
        np.asarray(model.n_categories, dtype=np.intp), (model.n_items,)
    )
    return [int(count) for count in counts]


def _item_coordinates(
    model: BaseItemModel,
    layouts: dict[str, _ParameterLayout],
) -> tuple[list[list[tuple[str, tuple[int, ...], int]]], int]:
    """Group free coordinates by item as (name, index within row, column).

    Coordinates of parameters shared by all items are appended to every
    item with their full storage index. Returns the groups and the number of
    shared coordinates.
    """
    per_item: list[list[tuple[str, tuple[int, ...], int]]] = [
        [] for _ in range(model.n_items)
    ]
    shared: list[tuple[str, tuple[int, ...], int]] = []
    offset = 0
    for name, layout in layouts.items():
        for rank, flat in enumerate(layout.free_indices):
            index = tuple(
                int(value) for value in np.unravel_index(int(flat), layout.shape)
            )
            if name in model._shared_parameters:
                shared.append((name, index, offset + rank))
            else:
                per_item[index[0]].append((name, index[1:], offset + rank))
        offset += layout.free_indices.size
    return [coordinates + shared for coordinates in per_item], len(shared)


def _log_terms_from_probabilities(
    probabilities: NDArray[np.float64],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert probability derivatives to log-probability derivatives."""
    active = (probabilities > PROB_EPSILON) & (probabilities < 1.0 - PROB_EPSILON)
    safe = np.where(active, probabilities, 1.0)
    log_first = np.where(active, first / safe, 0.0)
    log_second = second / safe - log_first[:, None] * log_first[None, :]
    return log_first, np.where(active, log_second, 0.0)


def _logistic_terms(
    model: BaseItemModel,
    item: int,
    theta: NDArray[np.float64],
    names: list[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    parameters = model._parameters
    a = float(parameters["discrimination"][item])
    b = float(parameters["difficulty"][item])
    c = float(parameters["guessing"][item]) if "guessing" in parameters else 0.0
    d = float(parameters["upper"][item]) if "upper" in parameters else 1.0
    centered = theta - b
    s = sigmoid(a * centered)
    u = s * (1.0 - s)
    v = u * (1.0 - 2.0 * s)
    span = d - c
    probability = c + span * s

    first = {
        "discrimination": span * u * centered,
        "difficulty": -span * u * a,
        "guessing": 1.0 - s,
        "upper": s,
    }
    second = {
        ("discrimination", "discrimination"): span * v * centered**2,
        ("discrimination", "difficulty"): -span * (u + a * centered * v),
        ("difficulty", "difficulty"): span * a * a * v,
        ("discrimination", "guessing"): -u * centered,
        ("discrimination", "upper"): u * centered,
        ("difficulty", "guessing"): u * a,
        ("difficulty", "upper"): -u * a,
    }
    k = len(names)
    sign = np.array([-1.0, 1.0])
    probabilities = np.column_stack((1.0 - probability, probability))
    d_prob = np.empty((k, theta.size, 2))
    d2_prob = np.zeros((k, k, theta.size, 2))
    for row, name in enumerate(names):
        d_prob[row] = first[name][:, None] * sign
        for column, other in enumerate(names):
            pair = second.get((name, other), second.get((other, name)))
            if pair is not None:
                d2_prob[row, column] = pair[:, None] * sign
    return _log_terms_from_probabilities(probabilities, d_prob, d2_prob)


def _graded_terms(
    a: float,
    thresholds: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return terms for (discrimination, thresholds 0..K-2) of a graded item."""
    n_categories = thresholds.size + 1
    centered = theta[:, None] - thresholds[None, :]
    s = sigmoid(a * centered)
    u = s * (1.0 - s)
    v = u * (1.0 - 2.0 * s)
    n_points = theta.size
    k = n_categories
    # Boundary curves P*_0 = 1 and P*_K = 0 pad the cumulative axis.
    cumulative = np.zeros((n_points, k + 1))
    cumulative[:, 0] = 1.0
    cumulative[:, 1:k] = s
    d_cum = np.zeros((k, n_points, k + 1))
    d2_cum = np.zeros((k, k, n_points, k + 1))
    columns = np.arange(1, k)
    d_cum[0, :, 1:k] = u * centered
    d2_cum[0, 0, :, 1:k] = v * centered**2
    for t, column in enumerate(columns):
        d_cum[1 + t, :, column] = -a * u[:, t]
        cross = -(u[:, t] + a * centered[:, t] * v[:, t])
        d2_cum[0, 1 + t, :, column] = cross
        d2_cum[1 + t, 0, :, column] = cross
        d2_cum[1 + t, 1 + t, :, column] = a * a * v[:, t]
    probabilities = cumulative[:, :-1] - cumulative[:, 1:]
    d_prob = d_cum[..., :-1] - d_cum[..., 1:]
    d2_prob = d2_cum[..., :-1] - d2_cum[..., 1:]
    return _log_terms_from_probabilities(probabilities, d_prob, d2_prob)


def _partial_credit_terms(
    a: float,
    steps: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return terms for (discrimination, steps 0..K-2) of a partial-credit item.

    Category logits are ``a * sum_{v < c} (theta - step_v)``, so their second
    derivatives are constant and the log-softmax derivatives follow directly.
    """
    n_categories = steps.size + 1
    n_points = theta.size
    k = n_categories
    features = np.zeros((n_points, k))
    features[:, 1:] = np.cumsum(theta[:, None] - steps[None, :], axis=1)
    logits = a * features
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    # Coordinate 1 + v moves every category above v.
    above = np.arange(k)[None, :] > np.arange(k - 1)[:, None]
    d_logit = np.empty((k, n_points, k))
    d_logit[0] = features
    d_logit[1:] = np.where(above, -a, 0.0)[:, None, :]
    d2_logit = np.zeros((k, k, 1, k))
    d2_logit[0, 1:, 0] = np.where(above, -1.0, 0.0)
    d2_logit[1:, 0, 0] = d2_logit[0, 1:, 0]

    mean = np.einsum("kqc,qc->kq", d_logit, probabilities)
    first = d_logit - mean[:, :, None]
    covariance = np.einsum("aqc,bqc,qc->abq", first, first, probabilities)
    mean_second = np.einsum("abc,qc->abq", d2_logit[:, :, 0], probabilities)
    second = d2_logit - (mean_second + covariance)[..., None]
    active = (probabilities > PROB_EPSILON) & (probabilities < 1.0 - PROB_EPSILON)
    return np.where(active, first, 0.0), np.where(active, second, 0.0)


def _analytic_terms(
    model: BaseItemModel,
    item: int,
    theta: NDArray[np.float64],
    coordinates: list[tuple[str, tuple[int, ...], int]],
    n_categories: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    from mirt.models.polytomous import (
        GradedRatingScaleModel,
        GradedResponseModel,
        RatingScaleModel,
    )

    if not model.is_polytomous:
        names = [name for name, _, _ in coordinates]
        return _logistic_terms(model, item, theta, names)
    parameters = model._parameters
    n_steps = n_categories - 1
    # Rating-scale items are graded or partial-credit items whose steps are
    # the item location plus the shared thresholds.
    if type(model) in (RatingScaleModel, GradedRatingScaleModel):
        steps = parameters["difficulty"][item] + parameters["thresholds"][:n_steps]
        a = float(parameters.get("discrimination", np.ones(1))[0])
    else:
        name = "thresholds" if type(model) is GradedResponseModel else "steps"
        steps = parameters[name][item, :n_steps]
        a = float(parameters["discrimination"][item])
    if type(model) in (GradedResponseModel, GradedRatingScaleModel):
        first, second = _graded_terms(a, steps, theta)
    else:
        first, second = _partial_credit_terms(a, steps, theta)
    # Each coordinate is a linear combination of the (slope, steps) rows.
    jacobian = np.zeros((len(coordinates), n_categories))
    for row, (name, index, _) in enumerate(coordinates):
        if name == "discrimination":
            jacobian[row, 0] = 1.0
        elif name == "difficulty":
            jacobian[row, 1:] = 1.0
        else:
            jacobian[row, 1 + index[0]] = 1.0
    k = len(coordinates)
    first = np.tensordot(jacobian, first, axes=1)
    second = np.tensordot(jacobian, second, axes=1).reshape(k, n_categories, -1)
    second = np.matmul(jacobian, second).reshape(k, k, *first.shape[1:])
    return first, second


def _numerical_terms(
    model: BaseItemModel,
    item: int,
    nodes: NDArray[np.float64],
    coordinates: list[tuple[str, tuple[int, ...], int]],
    n_categories: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Central differences of one item's clipped category log-probabilities."""
    names = {name for name, _, _ in coordinates}
    original = {name: model._parameters[name] for name in names}
    # Shared coordinates carry their full storage index.
    keys = [
        index if name in model._shared_parameters else (item, *index)
        for name, index, _ in coordinates
    ]
    centers = np.array(
        [
            original[name][key]
            for (name, _, _), key in zip(coordinates, keys, strict=True)
        ],
        dtype=np.float64,
    )
    steps = _DERIVATIVE_STEP * np.maximum(1.0, np.abs(centers))

    def log_probability(
        offsets: dict[int, float],
    ) -> NDArray[np.float64]:
        updated = {name: values.copy() for name, values in original.items()}
        for position, offset in offsets.items():
            updated[coordinates[position][0]][keys[position]] += offset
        model._parameters.update(updated)
        probabilities = np.asarray(model.probability(nodes, item), dtype=np.float64)
        if probabilities.ndim == 1:
            probabilities = np.column_stack((1.0 - probabilities, probabilities))
        return np.log(
            np.clip(probabilities[:, :n_categories], PROB_EPSILON, 1.0 - PROB_EPSILON)
        )

    k = len(coordinates)
    try:
        center = log_probability({})
        first = np.empty((k, *center.shape))
        second = np.empty((k, k, *center.shape))
        for row in range(k):
            h = steps[row]
            plus = log_probability({row: h})
            minus = log_probability({row: -h})
            first[row] = (plus - minus) / (2.0 * h)
            second[row, row] = (plus - 2.0 * center + minus) / h**2
            for column in range(row + 1, k):
                g = steps[column]
                cross = (
                    log_probability({row: h, column: g})
                    - log_probability({row: h, column: -g})
                    - log_probability({row: -h, column: g})
                    + log_probability({row: -h, column: -g})
                ) / (4.0 * h * g)
                second[row, column] = cross
                second[column, row] = cross
    finally:
        model._parameters.update(original)
    return first, second


def item_terms(
    model: BaseItemModel,
    nodes: NDArray[np.float64],
    layouts: dict[str, _ParameterLayout],
) -> list[_ItemTerms | None]:
    """Return each item's log-probability derivatives, or None without free ones."""
    from mirt.models.mixed_format import MixedItemModel

    if isinstance(model, MixedItemModel):
        return _mixed_item_terms(model, nodes, layouts)
    analytic = has_analytic_item_derivatives(model)
    theta = nodes[:, 0] if analytic else nodes
    categories = _item_categories(model)
    grouped, n_shared = _item_coordinates(model, layouts)
    terms: list[_ItemTerms | None] = []
    for item, coordinates in enumerate(grouped):
        if not coordinates:
            terms.append(None)
            continue
        compute = _analytic_terms if analytic else _numerical_terms
        first, second = compute(model, item, theta, coordinates, categories[item])
        columns = np.array([column for _, _, column in coordinates], dtype=np.intp)
        terms.append(_ItemTerms(columns, first, second, n_shared))
    return terms


def _mixed_item_terms(
    model: MixedItemModel,
    nodes: NDArray[np.float64],
    layouts: dict[str, _ParameterLayout],
) -> list[_ItemTerms | None]:
    """Place each component's item terms at its test items and columns.

    Every component's coordinates keep their columns in the mixed-format
    layout. Cross-component information then comes from the posterior
    covariance of the scores, as for any two items.
    """
    parts: dict[int, dict[str, _ParameterLayout]] = {}
    columns: dict[int, list[NDArray[np.intp]]] = {}
    offset = 0
    for name, layout in layouts.items():
        component, local = model.parameter_component(name)
        size = layout.free_indices.size
        parts.setdefault(component, {})[local] = layout
        columns.setdefault(component, []).append(np.arange(offset, offset + size))
        offset += size

    terms: list[_ItemTerms | None] = [None] * model.n_items
    components = model.components
    for component, local_layouts in parts.items():
        part, items = components[component]
        # Component columns follow ``local_layouts``; map them to the model's.
        placement = np.concatenate(columns[component])
        for position, term in enumerate(item_terms(part, nodes, local_layouts)):
            if term is not None:
                terms[items[position]] = _ItemTerms(
                    placement[term.columns], term.first, term.second
                )
    return terms


def _log_prior(prior_mass: NDArray[np.float64]) -> NDArray[np.float64]:
    log_mass = np.full(prior_mass.shape, -np.inf, dtype=np.float64)
    positive = prior_mass > 0.0
    log_mass[positive] = np.log(prior_mass[positive])
    return log_mass


def _posterior_block(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    nodes: NDArray[np.float64],
    log_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    log_joint = np.array(
        model.log_likelihood_batch(responses, nodes), dtype=np.float64, copy=True
    )
    posterior, log_marginal = normalize_log_posterior(log_joint, log_mass)
    if not np.all(np.isfinite(log_marginal)):
        raise ValueError("marginal log likelihood must be finite")
    return posterior, log_marginal


class _Scores:
    """Complete-data score sums kept in item-major parameter order.

    Each item's own coordinates occupy consecutive columns. Coordinates
    shared by all items follow them once, and every item's derivatives with
    respect to them accumulate there.
    """

    def __init__(self, terms: list[_ItemTerms | None]) -> None:
        self.terms = terms
        self.items = [item for item, term in enumerate(terms) if term is not None]
        self.n_shared = terms[self.items[0]].n_shared if self.items else 0
        own = [
            terms[item].columns[: terms[item].columns.size - self.n_shared]
            for item in self.items
        ]
        chunks = [np.empty(0, dtype=np.intp)]
        if own:
            chunks = [*own, terms[self.items[-1]].columns[own[-1].size :]]
        self.order = np.concatenate(chunks)
        self.n_parameters = int(self.order.size)
        self.offsets: list[int] = []
        offset = 0
        for columns in own:
            self.offsets.append(offset)
            offset += columns.size
        self.shared_start = offset

    def to_flat(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reorder an item-major matrix to the free-parameter layout."""
        result = np.zeros((self.n_parameters, self.n_parameters))
        result[np.ix_(self.order, self.order)] = matrix
        return result

    def item_columns(self, position: int) -> NDArray[np.intp]:
        """Return item-major columns of one item's own and shared coordinates."""
        start = self.offsets[position]
        own = self.terms[self.items[position]].columns.size - self.n_shared
        return np.r_[start : start + own, self.shared_start : self.n_parameters]

    def complete(self, counts: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Posterior-expected complete-data information from category counts."""
        result = np.zeros((self.n_parameters, self.n_parameters))
        for position, item in enumerate(self.items):
            second = self.terms[item].second
            block = np.ix_(self.item_columns(position), self.item_columns(position))
            result[block] -= np.einsum("abqc,qc->ab", second, counts[position])
        return result


class _CategoricalScores(_Scores):
    """Materialize person-by-node scores in bounded person blocks."""

    def __init__(self, terms: list[_ItemTerms | None]) -> None:
        super().__init__(terms)
        # Category-major tables with a trailing zero row for missing or
        # out-of-range responses, so one gather fills each item's columns.
        self.tables = []
        for item in self.items:
            first = terms[item].first
            table = np.zeros((first.shape[2] + 1, first.shape[1], first.shape[0]))
            table[:-1] = first.transpose(2, 1, 0)
            self.tables.append(table)
        n_points = self.tables[0].shape[1] if self.tables else 1
        self.variance = np.zeros((self.n_parameters, self.n_parameters))
        self.counts = [np.zeros((n_points, t.shape[0] - 1)) for t in self.tables]

    def block_size(self, n_points: int) -> int:
        return max(1, _MAX_BLOCK_ENTRIES // max(1, n_points * self.n_parameters))

    def _codes(self, values: NDArray[np.int_], n_categories: int) -> NDArray[np.intp]:
        valid = (values >= 0) & (values < n_categories)
        return np.where(valid, values, n_categories).astype(np.intp)

    def accumulate(
        self,
        responses: NDArray[np.int_],
        posterior: NDArray[np.float64],
        weights: NDArray[np.float64],
        observed: bool,
    ) -> NDArray[np.float64]:
        n_points = posterior.shape[1]
        scores = np.empty((responses.shape[0], n_points, self.n_parameters))
        scores[:, :, self.shared_start :] = 0.0
        own = slice(None, None if not self.n_shared else -self.n_shared)
        codes = []
        for position, item in enumerate(self.items):
            table = self.tables[position]
            item_codes = self._codes(responses[:, item], table.shape[0] - 1)
            start = self.offsets[position]
            gathered = table[item_codes]
            scores[:, :, start : start + table.shape[2] - self.n_shared] = gathered[
                ..., own
            ]
            if self.n_shared:
                scores[:, :, self.shared_start :] += gathered[..., own.stop :]
            codes.append(item_codes)
        person = np.matmul(posterior[:, None, :], scores)[:, 0, :]
        if observed:
            weighted = posterior * weights[:, None]
            scores *= np.sqrt(weighted)[:, :, None]
            flat = scores.reshape(-1, self.n_parameters)
            self.variance += flat.T @ flat
            for position, item_codes in enumerate(codes):
                n_categories = self.counts[position].shape[1]
                indicators = np.zeros((item_codes.size, n_categories + 1))
                indicators[np.arange(item_codes.size), item_codes] = 1.0
                self.counts[position] += weighted.T @ indicators[:, :-1]
        return person

    def missing_information(self) -> NDArray[np.float64]:
        return self.variance

    def complete_information(self) -> NDArray[np.float64]:
        return self.complete(self.counts)


class _BinaryScores(_Scores):
    """Factor dichotomous scores to avoid person-by-node-by-parameter arrays.

    For a binary item, ``d log P(0) = -P / (1 - P) * d log P(1)``, so each
    person's complete-data score at a node is a scalar response factor times
    one item-level vector. The posterior variance term then needs only an
    item-by-item Gram matrix per node.
    """

    def __init__(
        self,
        model: BaseItemModel,
        terms: list[_ItemTerms | None],
        nodes: NDArray[np.float64],
    ) -> None:
        super().__init__(terms)
        n_points = nodes.shape[0]
        sizes = [terms[item].columns.size for item in self.items]
        width = max(sizes, default=1)
        self.vectors = np.zeros((len(self.items), n_points, width))
        valid = np.zeros((len(self.items), width), dtype=np.bool_)
        for position, item in enumerate(self.items):
            first = terms[item].first
            self.vectors[position, :, : sizes[position]] = first[:, :, 1].T
            valid[position, : sizes[position]] = True
        self.flat = np.flatnonzero(valid.ravel())
        probability = np.asarray(model.probability(nodes), dtype=np.float64)
        probability = np.clip(
            probability.reshape(n_points, -1)[:, self.items],
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        # Clipped cells already carry zero derivatives in ``vectors``.
        self.ratio = -(probability / (1.0 - probability))
        self.gram = np.zeros((n_points, len(self.items), len(self.items)))
        self.correct = np.zeros((n_points, len(self.items)))
        self.incorrect = np.zeros_like(self.correct)

    def block_size(self, n_points: int) -> int:
        return max(1, _MAX_BLOCK_ENTRIES // max(1, n_points * len(self.items)))

    def accumulate(
        self,
        responses: NDArray[np.int_],
        posterior: NDArray[np.float64],
        weights: NDArray[np.float64],
        observed: bool,
    ) -> NDArray[np.float64]:
        values = responses[:, self.items]
        correct = (values == 1).astype(np.float64)
        incorrect = (values == 0).astype(np.float64)
        # factors[q, i, j]: 1 for a correct response, -odds for an incorrect one.
        factors = incorrect[None, :, :] * self.ratio[:, None, :]
        factors += correct[None, :, :]
        weighted_factors = factors * posterior.T[:, :, None]
        person = np.matmul(
            weighted_factors.transpose(2, 1, 0), self.vectors
        )  # (items, persons, width)
        person = person.transpose(1, 0, 2).reshape(responses.shape[0], -1)
        if observed:
            weighted = posterior * weights[:, None]
            factors *= np.sqrt(weighted).T[:, :, None]
            self.gram += np.matmul(factors.transpose(0, 2, 1), factors)
            self.correct += weighted.T @ correct
            self.incorrect += weighted.T @ incorrect
        return person[:, self.flat]

    def missing_information(self) -> NDArray[np.float64]:
        width = self.vectors.shape[2]
        size = len(self.items) * width
        variance = np.einsum(
            "jqa,lqb,qjl->jalb", self.vectors, self.vectors, self.gram, optimize=True
        ).reshape(size, size)
        return variance[np.ix_(self.flat, self.flat)]

    def complete_information(self) -> NDArray[np.float64]:
        counts = [
            np.column_stack((self.incorrect[:, position], self.correct[:, position]))
            for position in range(len(self.items))
        ]
        return self.complete(counts)


def _score_accumulator(
    model: BaseItemModel,
    terms: list[_ItemTerms | None],
    nodes: NDArray[np.float64],
) -> _CategoricalScores | _BinaryScores:
    if model.is_polytomous:
        return _CategoricalScores(terms)
    return _BinaryScores(model, terms, nodes)


def louis_information(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    nodes: NDArray[np.float64],
    prior_mass: NDArray[np.float64],
    layouts: dict[str, _ParameterLayout],
    *,
    person_weights: NDArray[np.float64] | None = None,
    meat_weights: NDArray[np.float64] | None = None,
    observed: bool = True,
) -> LouisInformation:
    """Accumulate exact observed information and score cross-products.

    Parameters
    ----------
    model : BaseItemModel
        Model accepted by :func:`supports_louis_information`.
    responses : ndarray of shape (n_persons, n_items)
        Responses with negative codes for missing cells.
    nodes : ndarray of shape (n_points, n_factors)
        Quadrature nodes.
    prior_mass : ndarray of shape (n_points,)
        Normalized prior mass at the nodes. Posteriors are recomputed from it.
    layouts : dict
        Free-parameter layout from ``standard_errors._flatten_parameters``.
    person_weights : ndarray of shape (n_persons,), optional
        Frequency or survey weights multiplying each person's information.
    meat_weights : ndarray of shape (n_persons,), optional
        Weights for the score cross-product; defaults to ``person_weights``.
    observed : bool, default=True
        Whether to accumulate the observed information. Score cross-products
        alone skip the dominant posterior variance products.

    Returns
    -------
    LouisInformation
        Information and score cross-product in the free-parameter layout.
    """
    terms = item_terms(model, nodes, layouts)
    scores = _score_accumulator(model, terms, nodes)
    n_persons = responses.shape[0]
    weights = (
        np.ones(n_persons, dtype=np.float64)
        if person_weights is None
        else np.asarray(person_weights, dtype=np.float64)
    )
    meat = None if meat_weights is None else np.asarray(meat_weights, np.float64)
    log_mass = _log_prior(prior_mass)

    score_outer = np.zeros((scores.n_parameters, scores.n_parameters))
    meat_outer = None if meat is None else np.zeros_like(score_outer)
    step = scores.block_size(nodes.shape[0])
    for start in range(0, n_persons, step):
        stop = min(start + step, n_persons)
        rows = responses[start:stop]
        posterior, _ = _posterior_block(model, rows, nodes, log_mass)
        block_weights = weights[start:stop]
        person = scores.accumulate(rows, posterior, block_weights, observed)
        score_outer += (person * block_weights[:, None]).T @ person
        if meat_outer is not None:
            meat_outer += (person * meat[start:stop, None]).T @ person

    crossproduct = scores.to_flat(score_outer if meat_outer is None else meat_outer)
    if not observed:
        return LouisInformation(np.zeros_like(crossproduct), crossproduct)
    information = scores.to_flat(
        scores.complete_information() - (scores.missing_information() - score_outer)
    )
    return LouisInformation((information + information.T) / 2.0, crossproduct)


def enumerated_patterns(model: BaseItemModel) -> NDArray[np.int_] | None:
    """Return every complete response pattern, or None beyond the bound."""
    categories = _item_categories(model)
    total = 1
    for count in categories:
        total *= count
        if total > MAX_ENUMERATED_PATTERNS:
            return None
    return np.array(list(product(*(range(count) for count in categories))))


def expected_louis_information(
    model: BaseItemModel,
    nodes: NDArray[np.float64],
    prior_mass: NDArray[np.float64],
    layouts: dict[str, _ParameterLayout],
    patterns: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Return the per-person expected information ``sum_y P(y) s(y) s(y)'``."""
    terms = item_terms(model, nodes, layouts)
    scores = _score_accumulator(model, terms, nodes)
    log_mass = _log_prior(prior_mass)
    information = np.zeros((scores.n_parameters, scores.n_parameters))
    step = scores.block_size(nodes.shape[0])
    for start in range(0, patterns.shape[0], step):
        rows = patterns[start : start + step]
        posterior, log_marginal = _posterior_block(model, rows, nodes, log_mass)
        person = scores.accumulate(rows, posterior, np.ones(rows.shape[0]), False)
        information += (person * np.exp(log_marginal)[:, None]).T @ person
    return scores.to_flat(information)
