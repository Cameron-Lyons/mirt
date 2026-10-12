"""Unconstrained latent class analysis of categorical items.

A latent class model assumes that respondents belong to one of ``K``
unobserved classes and that item responses are independent within a class.
Each class ``k`` has a proportion ``pi_k`` and, for every item ``j``, its own
probabilities ``P(X_j = c | k)`` of the categories ``c = 0, ..., C_j - 1``.
This is the model of R mirt's ``mdirt()`` with its default
``itemtype = "lca"``; :func:`fit_lca` (alias :func:`mdirt`) estimates it by
expectation maximization.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtDataError, MirtValidationError

# Probabilities are floored here before taking logarithms, so that a category
# whose estimate reaches zero contributes a large finite penalty instead of
# ``-inf``. The floor is far below any estimate that affects a likelihood.
_LOG_FLOOR = 1e-300
# Category probabilities and class proportions at or below this value (or at
# or above one minus it) are treated as boundary estimates: their logits
# diverge, so they are held fixed in the observed information and receive
# ``NaN`` standard errors.
_BOUNDARY_PROBABILITY = 1e-8
# Tolerance for probabilities that should sum to one.
_SUM_TOLERANCE = 1e-6
# Smallest eigenvalue of the unit-diagonal observed information below which
# the parameters are treated as not identified.
_IDENTIFICATION_TOLERANCE = 1e-10


def _is_integer(value: object) -> bool:
    """Return whether ``value`` is an integer other than a boolean."""
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _validate_count(value: object, name: str, minimum: int) -> int:
    """Return an integer control after checking its type and lower bound."""
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise MirtValidationError(
            f"{name} must be an integer", parameter=name, value=value
        )
    count = int(value)
    if count < minimum:
        raise MirtValidationError(
            f"{name} must be at least {minimum}",
            parameter=name,
            value=value,
            expected=f">= {minimum}",
        )
    return count


def _validate_bool(value: object, name: str) -> bool:
    """Return a boolean control after checking its type."""
    if not isinstance(value, (bool, np.bool_)):
        raise MirtValidationError(f"{name} must be boolean", parameter=name)
    return bool(value)


def _validate_item_names(item_names: Sequence[str] | None, n_items: int) -> list[str]:
    """Return item names, defaulting to ``Item_1, Item_2, ...``."""
    if item_names is None:
        return [f"Item_{index + 1}" for index in range(n_items)]
    if isinstance(item_names, (str, bytes)):
        raise MirtValidationError(
            "item_names must be a sequence of strings", parameter="item_names"
        )
    names = [str(name) for name in item_names]
    if len(names) != n_items:
        raise MirtValidationError(
            f"item_names has {len(names)} names, expected {n_items}",
            parameter="item_names",
        )
    if len(set(names)) != len(names):
        raise MirtValidationError("item_names must be unique", parameter="item_names")
    return names


def _validate_probability_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return a finite, non-negative probability vector that sums to one."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "biuf":
        raise MirtValidationError(f"{name} must be numeric", parameter=name)
    vector = np.array(raw, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise MirtValidationError(
            f"{name} must be a non-empty one-dimensional array", parameter=name
        )
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise MirtValidationError(
            f"{name} must contain finite non-negative values", parameter=name
        )
    total = float(np.sum(vector))
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise MirtValidationError(
            f"{name} must sum to one, got {total:.6g}", parameter=name
        )
    return vector / total


class LatentClassModel:
    """Latent class model with unconstrained item category probabilities.

    Respondents belong to one of ``n_classes`` latent classes with
    probabilities :attr:`class_proportions`. Within class ``k`` the items are
    independent and item ``j`` takes category ``c`` with probability
    ``item_probabilities[j][k, c]``. Responses are coded ``0, ...,
    n_categories[j] - 1``; negative values and ``NaN`` are missing and leave
    the likelihood unchanged.

    :func:`fit_lca` returns a fitted model inside an :class:`LCAResult`. A
    model can also be built from known parameters, for example to simulate
    data with :meth:`simulate`.

    Parameters
    ----------
    class_proportions : array-like of shape (n_classes,)
        Latent class probabilities; non-negative and summing to one.
    item_probabilities : sequence of array-like
        One array of shape ``(n_classes, n_categories_j)`` per item with at
        least two categories. Row ``k`` holds the category probabilities of
        class ``k`` and must sum to one.
    item_names : sequence of str, optional
        Unique item names. Defaults to ``Item_1, Item_2, ...``.

    Examples
    --------
    >>> import numpy as np
    >>> from mirt.models.latent_class import LatentClassModel
    >>> model = LatentClassModel(
    ...     [0.6, 0.4],
    ...     [np.array([[0.2, 0.8], [0.9, 0.1]])] * 4,
    ... )
    >>> responses, classes = model.simulate(500, seed=1)
    >>> model.posterior(responses[:2]).shape
    (2, 2)
    """

    def __init__(
        self,
        class_proportions: ArrayLike,
        item_probabilities: Sequence[ArrayLike],
        item_names: Sequence[str] | None = None,
    ) -> None:
        proportions = _validate_probability_vector(
            class_proportions, "class_proportions"
        )
        n_classes = proportions.size
        if isinstance(item_probabilities, np.ndarray) and item_probabilities.ndim == 3:
            item_probabilities = list(item_probabilities)
        if not isinstance(item_probabilities, Sequence) or not item_probabilities:
            raise MirtValidationError(
                "item_probabilities must be a non-empty sequence of arrays",
                parameter="item_probabilities",
            )
        tables: list[NDArray[np.float64]] = []
        for index, values in enumerate(item_probabilities):
            raw = np.asarray(values)
            if raw.dtype.kind not in "biuf":
                raise MirtValidationError(
                    f"item_probabilities[{index}] must be numeric",
                    parameter="item_probabilities",
                )
            table = np.array(raw, dtype=np.float64)
            if table.ndim != 2 or table.shape[0] != n_classes or table.shape[1] < 2:
                raise MirtValidationError(
                    f"item_probabilities[{index}] must have shape "
                    f"({n_classes}, n_categories) with at least two categories, "
                    f"got {table.shape}",
                    parameter="item_probabilities",
                )
            if not np.all(np.isfinite(table)) or np.any(table < 0.0):
                raise MirtValidationError(
                    f"item_probabilities[{index}] must contain finite "
                    "non-negative values",
                    parameter="item_probabilities",
                )
            totals = np.sum(table, axis=1)
            if np.any(np.abs(totals - 1.0) > _SUM_TOLERANCE):
                raise MirtValidationError(
                    f"every row of item_probabilities[{index}] must sum to one",
                    parameter="item_probabilities",
                )
            tables.append(table / totals[:, None])

        self._class_proportions = proportions
        self._item_probabilities = tables
        self._n_categories = np.array([table.shape[1] for table in tables])
        self._item_names = _validate_item_names(item_names, len(tables))
        self._offsets = np.concatenate([[0], np.cumsum(self._n_categories)])

    @property
    def n_classes(self) -> int:
        """Number of latent classes."""
        return int(self._class_proportions.size)

    @property
    def n_items(self) -> int:
        """Number of items."""
        return len(self._item_probabilities)

    @property
    def n_categories(self) -> NDArray[np.int_]:
        """Number of response categories of every item."""
        return self._n_categories.copy()

    @property
    def item_names(self) -> list[str]:
        """Item names in column order."""
        return list(self._item_names)

    @property
    def class_proportions(self) -> NDArray[np.float64]:
        """Latent class probabilities, shape ``(n_classes,)``."""
        return self._class_proportions.copy()

    @property
    def item_probabilities(self) -> list[NDArray[np.float64]]:
        """Category probabilities per item, each of shape ``(n_classes, C_j)``."""
        return [table.copy() for table in self._item_probabilities]

    @property
    def model_name(self) -> str:
        """Short model label, such as ``"LCA(3)"`` for three classes."""
        return f"LCA({self.n_classes})"

    @property
    def n_parameters(self) -> int:
        """Free parameters: ``K - 1 + K * sum_j (C_j - 1)`` for ``K`` classes."""
        return int(self.n_classes - 1 + self.n_classes * np.sum(self._n_categories - 1))

    def _flat_probabilities(self) -> NDArray[np.float64]:
        """Return the category probabilities as one ``(K, sum_j C_j)`` array."""
        return np.concatenate(self._item_probabilities, axis=1)

    def _validate_new_responses(self, responses: Any) -> NDArray[np.int_]:
        """Validate responses to score, aligning named columns to the items."""
        from mirt.utils.data import align_response_columns, validate_responses

        aligned = align_response_columns(responses, self._item_names)
        values = validate_responses(aligned, n_items=self.n_items)
        _check_category_codes(values, self._n_categories)
        return values

    def class_log_likelihoods(self, responses: Any) -> NDArray[np.float64]:
        """Log-likelihood of every response row under every class.

        Parameters
        ----------
        responses : array-like or DataFrame of shape (n_persons, n_items)
            Responses coded ``0, ..., n_categories[j] - 1``. Negative values
            and ``NaN`` are missing. DataFrame columns are matched to
            :attr:`item_names` by label.

        Returns
        -------
        ndarray of shape (n_persons, n_classes)
            ``log P(x_i | class k)``; rows without observed responses are 0.
        """
        values = self._validate_new_responses(responses)
        indicators = _category_indicators(values, self._offsets)
        log_probabilities = np.log(np.maximum(self._flat_probabilities(), _LOG_FLOOR))
        return indicators @ log_probabilities.T

    def posterior(self, responses: Any) -> NDArray[np.float64]:
        """Posterior class probabilities of response rows.

        The class proportions are the prior. Rows without observed responses
        receive the class proportions.

        Parameters
        ----------
        responses : array-like or DataFrame of shape (n_persons, n_items)
            Responses, as for :meth:`class_log_likelihoods`.

        Returns
        -------
        ndarray of shape (n_persons, n_classes)
            Posterior probabilities; every row sums to one.
        """
        posterior, _ = _class_posterior(
            self.class_log_likelihoods(responses), self._class_proportions
        )
        return posterior

    def classify(self, responses: Any) -> NDArray[np.int_]:
        """Assign every response row to its most probable class (MAP).

        Ties go to the lower class index.

        Parameters
        ----------
        responses : array-like or DataFrame of shape (n_persons, n_items)
            Responses, as for :meth:`class_log_likelihoods`.

        Returns
        -------
        ndarray of shape (n_persons,)
            Class indices in ``0, ..., n_classes - 1``.
        """
        return np.argmax(self.posterior(responses), axis=1).astype(np.int_)

    def log_likelihood(self, responses: Any) -> float:
        """Marginal log-likelihood of response rows summed over respondents.

        Parameters
        ----------
        responses : array-like or DataFrame of shape (n_persons, n_items)
            Responses, as for :meth:`class_log_likelihoods`.

        Returns
        -------
        float
            ``sum_i log sum_k pi_k P(x_i | k)``.
        """
        _, marginal = _class_posterior(
            self.class_log_likelihoods(responses), self._class_proportions
        )
        return float(np.sum(marginal))

    def simulate(
        self, n_persons: int, seed: int | np.random.Generator | None = None
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Simulate complete response patterns.

        Parameters
        ----------
        n_persons : int
            Number of respondents, at least one.
        seed : int or numpy.random.Generator, optional
            Seed or generator for reproducible draws.

        Returns
        -------
        tuple
            ``(responses, classes)``: integer responses of shape
            ``(n_persons, n_items)`` and the class index of every respondent.
        """
        n_persons = _validate_count(n_persons, "n_persons", 1)
        rng = np.random.default_rng(seed)
        classes = rng.choice(self.n_classes, size=n_persons, p=self._class_proportions)
        responses = np.empty((n_persons, self.n_items), dtype=np.int_)
        uniforms = rng.random((n_persons, self.n_items))
        for item, table in enumerate(self._item_probabilities):
            cumulative = np.cumsum(table, axis=1)[classes, :-1]
            responses[:, item] = np.sum(uniforms[:, item, None] >= cumulative, axis=1)
        return responses, classes.astype(np.int_)


@dataclass
class LCAResult:
    """Result of fitting a latent class model with :func:`fit_lca`.

    Classes are ordered by decreasing estimated proportion, so class 0 is
    the largest. Arrays indexed by class follow that order.

    Attributes
    ----------
    model : LatentClassModel
        Fitted model with the estimated class proportions and item category
        probabilities.
    log_likelihood : float
        Marginal log-likelihood at the returned estimates. With
        ``smoothing > 0`` it excludes the log-prior.
    n_parameters : int
        ``K - 1 + K * sum_j (C_j - 1)`` for ``K`` classes and ``C_j``
        categories of item ``j``.
    n_observations : int
        Number of respondents with at least one observed response, the
        sample size of :attr:`bic` and :attr:`sabic`.
    n_iterations : int
        EM iterations of the selected start.
    converged : bool
        Whether the selected start met ``tol`` before ``max_iter``
        iterations.
    class_posteriors : ndarray of shape (n_persons, n_classes)
        Posterior class probabilities of the fitted respondents, one row per
        input row.
    start_log_likelihoods : ndarray of shape (n_starts,)
        Final log-likelihood of every start, in the order they were run.
    entropy : float
        Relative entropy of the classification,
        ``1 - sum_i sum_k -w_ik log w_ik / (n log K)`` over the posterior
        probabilities ``w`` of the respondents with observed responses.
        It lies between 0 and 1, and values near 1 indicate well separated
        classes. ``NaN`` for one class.
    class_proportions_se : ndarray of shape (n_classes,) or None
        Observed-information standard errors of the class proportions, or
        ``None`` when ``compute_standard_errors=False``.
    item_probabilities_se : list of ndarray or None
        Standard errors matching :attr:`item_probabilities`, or ``None``.
        Boundary estimates (probabilities within ``1e-8`` of 0 or 1) have
        no Wald standard error and are ``NaN``.
    """

    model: LatentClassModel
    log_likelihood: float
    n_parameters: int
    n_observations: int
    n_iterations: int
    converged: bool
    class_posteriors: NDArray[np.float64]
    start_log_likelihoods: NDArray[np.float64]
    entropy: float
    class_proportions_se: NDArray[np.float64] | None = None
    item_probabilities_se: list[NDArray[np.float64]] | None = None

    @property
    def n_classes(self) -> int:
        """Number of latent classes."""
        return self.model.n_classes

    @property
    def class_proportions(self) -> NDArray[np.float64]:
        """Estimated class proportions, largest first."""
        return self.model.class_proportions

    @property
    def item_probabilities(self) -> list[NDArray[np.float64]]:
        """Estimated ``P(X_j = c | class k)``, one ``(K, C_j)`` array per item."""
        return self.model.item_probabilities

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return -2.0 * self.log_likelihood + 2.0 * self.n_parameters

    @property
    def bic(self) -> float:
        """Bayesian information criterion with the respondents as ``n``."""
        return -2.0 * self.log_likelihood + self.n_parameters * np.log(
            max(self.n_observations, 1)
        )

    @property
    def sabic(self) -> float:
        """Sample-size adjusted BIC, with ``n* = (n + 2) / 24``."""
        return -2.0 * self.log_likelihood + self.n_parameters * np.log(
            (max(self.n_observations, 1) + 2.0) / 24.0
        )

    def posterior(self, responses: Any = None) -> NDArray[np.float64]:
        """Posterior class probabilities of the fitted or of new respondents.

        Parameters
        ----------
        responses : array-like or DataFrame, optional
            New responses to score with the fitted model. ``None`` returns
            :attr:`class_posteriors`.

        Returns
        -------
        ndarray of shape (n_persons, n_classes)
            Posterior class probabilities.
        """
        if responses is None:
            return self.class_posteriors.copy()
        return self.model.posterior(responses)

    def classify(self, responses: Any = None) -> NDArray[np.int_]:
        """Most probable (MAP) class of the fitted or of new respondents.

        Parameters
        ----------
        responses : array-like or DataFrame, optional
            New responses to classify. ``None`` classifies the fitted data.

        Returns
        -------
        ndarray of shape (n_persons,)
            Class indices; ties go to the lower (larger) class.
        """
        return np.argmax(self.posterior(responses), axis=1).astype(np.int_)

    def summary(self) -> str:
        """Return a text summary of the fit and the estimated profiles."""
        width = 72
        lines = [
            "=" * width,
            f"{'Latent Class Analysis Results':^{width}}",
            "=" * width,
            f"Classes:            {self.n_classes}",
            f"Respondents:        {self.n_observations}",
            f"Parameters:         {self.n_parameters}",
            f"Log-Likelihood:     {self.log_likelihood:.4f}",
            f"AIC:                {self.aic:.4f}",
            f"BIC:                {self.bic:.4f}",
            f"SABIC:              {self.sabic:.4f}",
            f"Entropy:            {self.entropy:.4f}",
            f"Iterations:         {self.n_iterations}",
            f"Converged:          {self.converged}",
            f"Starts:             {self.start_log_likelihoods.size}",
            "-" * width,
        ]
        labels = [f"Class {k}" for k in range(self.n_classes)]
        name_width = max(12, *(len(name) + 4 for name in self.model.item_names))
        header = f"{'':<{name_width}}" + "".join(f"{label:>10}" for label in labels)
        lines.append(header)
        lines.append(
            f"{'Proportion':<{name_width}}"
            + "".join(f"{value:>10.3f}" for value in self.class_proportions)
        )
        lines.append("-" * width)
        lines.append("Item category probabilities:")
        for name, table in zip(
            self.model.item_names, self.item_probabilities, strict=True
        ):
            for category in range(table.shape[1]):
                label = f"{name} = {category}"
                lines.append(
                    f"{label:<{name_width}}"
                    + "".join(f"{value:>10.3f}" for value in table[:, category])
                )
        lines.append("=" * width)
        return "\n".join(lines)


def _check_category_codes(
    values: NDArray[np.int_], n_categories: NDArray[np.int_]
) -> None:
    """Reject response codes outside an item's categories."""
    too_large = values >= n_categories[None, :]
    if np.any(too_large):
        item = int(np.nonzero(np.any(too_large, axis=0))[0][0])
        raise MirtDataError(
            f"item {item} has response code {int(values[:, item].max())}, "
            f"but only {int(n_categories[item])} categories "
            f"(codes 0 to {int(n_categories[item]) - 1})",
            n_items=values.shape[1],
        )


def _category_indicators(
    values: NDArray[np.int_], offsets: NDArray[np.int_]
) -> NDArray[np.float64]:
    """One-hot response indicators, one column per item category.

    Missing responses have no indicator, so their rows contribute nothing.
    """
    indicators = np.zeros((values.shape[0], int(offsets[-1])), dtype=np.float64)
    rows, items = np.nonzero(values >= 0)
    indicators[rows, offsets[items] + values[rows, items]] = 1.0
    return indicators


def _class_posterior(
    class_log_likelihoods: NDArray[np.float64],
    class_proportions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return class posteriors and each row's marginal log-likelihood."""
    joint = class_log_likelihoods + np.log(np.maximum(class_proportions, _LOG_FLOOR))
    shift = np.max(joint, axis=1, keepdims=True)
    posterior = np.exp(joint - shift)
    totals = np.sum(posterior, axis=1, keepdims=True)
    posterior /= totals
    return posterior, (shift + np.log(totals))[:, 0]


def _relative_entropy(
    posterior: NDArray[np.float64], counts: NDArray[np.float64]
) -> float:
    """Relative entropy of count-weighted posterior class probabilities."""
    n_classes = posterior.shape[1]
    total = float(np.sum(counts))
    if n_classes == 1 or total == 0.0:
        return float("nan")
    terms = np.zeros_like(posterior)
    positive = posterior > 0.0
    terms[positive] = posterior[positive] * np.log(posterior[positive])
    information = -float(counts @ np.sum(terms, axis=1))
    return 1.0 - information / (total * np.log(n_classes))


def _infer_categories(
    values: NDArray[np.int_], n_categories: int | Sequence[int] | None
) -> NDArray[np.int_]:
    """Return per-item category counts, inferred or checked against the data."""
    n_items = values.shape[1]
    highest = np.max(values, axis=0)
    if n_categories is None:
        return np.maximum(highest + 1, 2).astype(np.int_)
    if isinstance(n_categories, (int, np.integer)):
        if isinstance(n_categories, (bool, np.bool_)):
            raise MirtValidationError(
                "n_categories must be an integer or a sequence of integers",
                parameter="n_categories",
            )
        counts = np.full(n_items, int(n_categories), dtype=np.int_)
    else:
        if isinstance(n_categories, (str, bytes)) or not isinstance(
            n_categories, Sequence
        ):
            raise MirtValidationError(
                "n_categories must be an integer or a sequence of integers",
                parameter="n_categories",
            )
        if len(n_categories) != n_items or not all(
            _is_integer(count) for count in n_categories
        ):
            raise MirtValidationError(
                f"n_categories must contain {n_items} integers",
                parameter="n_categories",
            )
        counts = np.array([int(count) for count in n_categories], dtype=np.int_)
    if np.any(counts < 2):
        raise MirtValidationError(
            "every item needs at least two categories", parameter="n_categories"
        )
    _check_category_codes(values, counts)
    return counts


def _starting_values(
    n_classes: int,
    n_categories: NDArray[np.int_],
    rng: np.random.Generator | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return class proportions and flat category probabilities for one start.

    Without ``rng`` these are R mirt's ``mdirt`` defaults: equal proportions
    and, for every item, the same logit ``L_k`` of each nonzero category
    against category 0 in class ``k``, with ``L`` evenly spaced from
    ``-1 - log K`` to ``1 + log K``. Random starts draw the proportions from
    a concentrated Dirichlet, which keeps every class large enough to
    estimate, and each class's category probabilities from a flat one.
    """
    if rng is None:
        proportions = np.full(n_classes, 1.0 / n_classes)
        spread = 1.0 + np.log(n_classes)
        logits = np.linspace(-spread, spread, n_classes)
        tables = []
        for count in n_categories:
            category_logits = np.zeros((n_classes, int(count)))
            category_logits[:, 1:] = logits[:, None]
            table = np.exp(category_logits)
            tables.append(table / np.sum(table, axis=1, keepdims=True))
    else:
        proportions = rng.dirichlet(np.full(n_classes, 4.0))
        tables = [
            rng.dirichlet(np.ones(int(count)), size=n_classes) for count in n_categories
        ]
    return proportions, np.concatenate(tables, axis=1)


@dataclass
class _EMState:
    """Estimates and diagnostics of one EM run on collapsed patterns."""

    class_proportions: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    posterior: NDArray[np.float64]
    log_likelihood: float
    objective: float
    converged: bool
    n_iterations: int


def _log_prior(probabilities: NDArray[np.float64], smoothing: float) -> float:
    """Dirichlet log-prior (up to a constant) of the category probabilities."""
    if smoothing == 0.0:
        return 0.0
    return smoothing * float(np.sum(np.log(np.maximum(probabilities, _LOG_FLOOR))))


def _run_em(
    indicators: NDArray[np.float64],
    counts: NDArray[np.float64],
    item_of_column: NDArray[np.int_],
    offsets: NDArray[np.int_],
    n_categories: NDArray[np.int_],
    class_proportions: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    *,
    smoothing: float,
    max_iter: int,
    tol: float,
    verbose: bool,
    label: str,
) -> _EMState:
    """Iterate EM from the given values until the objective settles.

    The objective is the marginal log-likelihood, plus the Dirichlet
    log-prior when ``smoothing > 0``. Iteration stops when its absolute
    change between consecutive iterations is below ``tol``.
    """
    n_total = float(np.sum(counts))
    column_categories = n_categories[item_of_column].astype(np.float64)

    def e_step(
        proportions: NDArray[np.float64], flat: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float]:
        log_probabilities = np.log(np.maximum(flat, _LOG_FLOOR))
        posterior, marginal = _class_posterior(
            indicators @ log_probabilities.T, proportions
        )
        return posterior, float(counts @ marginal)

    posterior, log_likelihood = e_step(class_proportions, probabilities)
    objective = log_likelihood + _log_prior(probabilities, smoothing)
    converged = False
    n_iterations = 0
    for iteration in range(1, max_iter + 1):
        weighted = posterior * counts[:, None]
        class_proportions = np.sum(weighted, axis=0) / n_total
        category_counts = (indicators.T @ weighted).T
        item_totals = np.add.reduceat(category_counts, offsets[:-1], axis=1)
        denominators = item_totals[:, item_of_column] + smoothing * column_categories
        # A class without observed responses to an item keeps its estimates.
        probabilities = np.divide(
            category_counts + smoothing,
            denominators,
            out=probabilities.copy(),
            where=denominators > 0.0,
        )
        posterior, log_likelihood = e_step(class_proportions, probabilities)
        next_objective = log_likelihood + _log_prior(probabilities, smoothing)
        change = next_objective - objective
        objective = next_objective
        n_iterations = iteration
        if verbose:
            print(
                f"{label}iteration {iteration}: LL = {log_likelihood:.6f}, "
                f"change = {change:.6g}"
            )
        if abs(change) < tol:
            converged = True
            break
    return _EMState(
        class_proportions=class_proportions,
        probabilities=probabilities,
        posterior=posterior,
        log_likelihood=log_likelihood,
        objective=objective,
        converged=converged,
        n_iterations=n_iterations,
    )


def _softmax_jacobian(
    probabilities: NDArray[np.float64], free: NDArray[np.int_]
) -> NDArray[np.float64]:
    """Jacobian of all probabilities with respect to the free logits."""
    jacobian = -np.outer(probabilities, probabilities[free])
    jacobian[free, np.arange(free.size)] += probabilities[free]
    return jacobian


def _standard_errors(
    indicators: NDArray[np.float64],
    observed: NDArray[np.float64],
    counts: NDArray[np.float64],
    posterior: NDArray[np.float64],
    class_proportions: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    item_of_column: NDArray[np.int_],
    offsets: NDArray[np.int_],
    n_categories: NDArray[np.int_],
    smoothing: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Observed-information standard errors of proportions and probabilities.

    The information is exact (Louis' identity) in the baseline-logit
    parameterization: class logits against the largest class, and within
    each item and class, category logits against the most probable category.
    Boundary estimates are held fixed. The standard errors are mapped to the
    probability scale by the delta method. Returns flat arrays matching
    ``class_proportions`` and ``probabilities``; boundary estimates are
    ``NaN``.
    """
    n_classes, n_columns = probabilities.shape
    lower, upper = _BOUNDARY_PROBABILITY, 1.0 - _BOUNDARY_PROBABILITY
    weighted = posterior * counts[:, None]
    n_total = float(np.sum(counts))

    reference_class = int(np.argmax(class_proportions))
    free_classes = np.array(
        [
            k
            for k in range(n_classes)
            if k != reference_class and class_proportions[k] > lower
        ],
        dtype=np.int_,
    )
    n_class_free = free_classes.size

    # Free item logits per class: all but the most probable category of each
    # item, excluding categories at the boundary.
    free_columns: list[NDArray[np.int_]] = []
    for k in range(n_classes):
        is_free = (probabilities[k] > lower) & (probabilities[k] < upper)
        for item in range(offsets.size - 1):
            start, stop = offsets[item], offsets[item + 1]
            is_free[start + int(np.argmax(probabilities[k, start:stop]))] = False
        free_columns.append(np.flatnonzero(is_free))
    sizes = [n_class_free, *(columns.size for columns in free_columns)]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    n_free = int(starts[-1])

    complete = np.zeros((n_free, n_free))
    expected_outer = np.zeros((n_free, n_free))
    mean_scores = np.zeros((counts.size, n_free))

    free_proportions = class_proportions[free_classes]
    class_block = slice(0, n_class_free)
    complete[class_block, class_block] = n_total * (
        np.diag(free_proportions) - np.outer(free_proportions, free_proportions)
    )
    mean_scores[:, class_block] = posterior[:, free_classes] - free_proportions
    class_scores = np.zeros((n_classes, n_class_free))
    class_scores[free_classes, np.arange(n_class_free)] = 1.0
    class_scores -= free_proportions
    class_totals = np.sum(weighted, axis=0)
    expected_outer[class_block, class_block] = (
        class_scores.T * class_totals
    ) @ class_scores

    for k in range(n_classes):
        columns = free_columns[k]
        if columns.size == 0:
            continue
        block = slice(int(starts[k + 1]), int(starts[k + 2]))
        items = item_of_column[columns]
        values = probabilities[k, columns]
        scores = indicators[:, columns] - observed[:, items] * values
        item_counts = weighted[:, k] @ observed[:, items]
        scale = item_counts + smoothing * n_categories[items]
        same_item = items[:, None] == items[None, :]
        complete[block, block] = np.diag(scale * values) - same_item * (
            scale[:, None] * np.outer(values, values)
        )
        expected_outer[block, block] = (scores.T * weighted[:, k]) @ scores
        expected_outer[class_block, block] = np.outer(
            class_scores[k], weighted[:, k] @ scores
        )
        expected_outer[block, class_block] = expected_outer[class_block, block].T
        mean_scores[:, block] = posterior[:, k, None] * scores

    information = complete - expected_outer + (mean_scores.T * counts) @ mean_scores
    information = 0.5 * (information + information.T)
    # Invert the information scaled to a unit diagonal, whose smallest
    # eigenvalue measures collinearity regardless of how little information
    # a near-boundary estimate carries.
    covariance = np.full((n_free, n_free), np.nan)
    diagonal = np.diag(information)
    if n_free and np.all(diagonal > 0.0):
        scale = np.sqrt(diagonal)
        scaled = information / np.outer(scale, scale)
        if np.linalg.eigvalsh(scaled)[0] > _IDENTIFICATION_TOLERANCE:
            covariance = np.linalg.inv(scaled) / np.outer(scale, scale)
    if n_free and not np.all(np.isfinite(covariance)):
        warnings.warn(
            "the observed information of the latent class model is singular, "
            "so standard errors are unavailable; the model may have more "
            "classes than the data identify",
            RuntimeWarning,
            stacklevel=3,
        )

    proportion_se = np.full(n_classes, np.nan)
    jacobian = _softmax_jacobian(class_proportions, free_classes)
    variances = np.einsum(
        "ij,jk,ik->i", jacobian, covariance[class_block, class_block], jacobian
    )
    estimable = (class_proportions > lower) & (class_proportions < upper)
    proportion_se[estimable] = np.sqrt(np.maximum(variances[estimable], 0.0))
    if n_classes == 1:
        # A single class has proportion one by definition.
        proportion_se[:] = 0.0

    probability_se = np.full((n_classes, n_columns), np.nan)
    for k in range(n_classes):
        block = slice(int(starts[k + 1]), int(starts[k + 2]))
        class_covariance = covariance[block, block]
        for item in range(offsets.size - 1):
            start, stop = int(offsets[item]), int(offsets[item + 1])
            in_item = (free_columns[k] >= start) & (free_columns[k] < stop)
            local_free = free_columns[k][in_item] - start
            item_probabilities = probabilities[k, start:stop]
            jacobian = _softmax_jacobian(item_probabilities, local_free)
            item_covariance = class_covariance[np.ix_(in_item, in_item)]
            variances = np.einsum("ij,jk,ik->i", jacobian, item_covariance, jacobian)
            estimable = (item_probabilities > lower) & (item_probabilities < upper)
            item_se = probability_se[k, start:stop]
            item_se[estimable] = np.sqrt(np.maximum(variances[estimable], 0.0))
    return proportion_se, probability_se


def fit_lca(
    data: Any,
    n_classes: int,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    *,
    n_starts: int = 1,
    seed: int | None = None,
    smoothing: float = 0.0,
    n_categories: int | Sequence[int] | None = None,
    item_names: Sequence[str] | None = None,
    compute_standard_errors: bool = True,
) -> LCAResult:
    """Fit an unconstrained latent class model by expectation maximization.

    Each of ``n_classes`` latent classes has a proportion and, for every
    item, its own category probabilities; responses are independent within
    a class. This is the model of R mirt's ``mdirt(data, n_classes)`` with
    its default ``itemtype = "lca"``, for dichotomous and polytomous
    (nominal) items. The E-step computes posterior class probabilities of
    the distinct response patterns and the M-step updates the proportions
    and probabilities in closed form.

    Parameters
    ----------
    data : array-like or DataFrame of shape (n_persons, n_items)
        Responses coded ``0, ..., C_j - 1``. Negative values and ``NaN``
        (including pandas and polars nulls) are missing; they leave the
        likelihood unchanged. Rows without observed responses carry no
        information: they receive the class proportions as their posterior
        and do not count as observations, as in R mirt. Every item needs at
        least one observed response.
    n_classes : int
        Number of latent classes, at least one.
    max_iter : int, default=500
        Maximum number of EM iterations per start.
    tol : float, default=1e-4
        Convergence tolerance on the absolute change in the marginal
        log-likelihood (log-posterior when ``smoothing > 0``) between
        consecutive iterations.
    verbose : bool, default=False
        Print the log-likelihood after every iteration.
    n_starts : int, default=1
        Number of starts. The first uses R mirt's deterministic ``mdirt``
        starting values; later starts draw random proportions and category
        probabilities. The start with the largest final log-likelihood
        (log-posterior when ``smoothing > 0``) is returned. Latent class
        likelihoods often have local maxima, so several starts are
        advisable, like ``nruns`` in R.
    seed : int, optional
        Seed for the random starts.
    smoothing : float, default=0.0
        Non-negative pseudo-count added to every category count of every
        class in the M-step: the estimates are then posterior modes under a
        symmetric Dirichlet prior with concentration ``1 + smoothing`` on the
        category probabilities of each item and class. The default 0 gives
        maximum-likelihood estimates, as R mirt's ``mdirt``, which places no
        prior on its ``lca`` items; a small value such as 0.5 keeps the
        probabilities away from 0 and 1.
    n_categories : int or sequence of int, optional
        Category count of every item, or one count per item, at least 2.
        Inferred by default from each item's largest observed code, with a
        minimum of two. A category that no respondent uses is still counted
        and gets probability 0 (or its smoothed value); recode the
        responses to consecutive codes to drop it, as R does.
    item_names : sequence of str, optional
        Item names. Defaults to unique DataFrame column names when present,
        otherwise ``Item_1, Item_2, ...``.
    compute_standard_errors : bool, default=True
        Compute observed-information standard errors of the class
        proportions and item category probabilities.

    Returns
    -------
    LCAResult
        Fitted :class:`LatentClassModel`, log-likelihood, AIC, BIC, SABIC,
        relative entropy, convergence information, posterior class
        probabilities of the respondents and, optionally, standard errors.

    Raises
    ------
    MirtValidationError
        If a control argument is invalid.
    MirtDataError
        If the responses are not a valid response matrix, an item has no
        observed responses, or a code exceeds an item's categories.

    Notes
    -----
    Latent classes are unordered, so their labels are arbitrary. The
    returned classes are ordered by decreasing estimated proportion, so
    class 0 is the largest; ties keep the order in which EM found them. R
    mirt instead keeps the order of its starting values, so its class
    numbers may differ by a permutation.

    R's starting values give every nonzero category of an item the same
    logit, which suits ordered binary data but can leave polytomous fits in
    a poor local maximum. Compare :attr:`LCAResult.start_log_likelihoods`
    over several starts: a best value reached by several starts is more
    likely the global maximum.

    The number of free parameters is ``K - 1 + K * sum_j (C_j - 1)``. A model
    with more classes than the data identify has a singular information
    matrix; its standard errors are then ``NaN`` with a ``RuntimeWarning``.

    Examples
    --------
    >>> import mirt
    >>> responses = mirt.load_dataset("LSAT7")["data"]
    >>> result = mirt.fit_lca(responses, n_classes=2)
    >>> round(result.log_likelihood, 1)
    -2660.3
    >>> result.class_proportions.round(1)
    array([0.7, 0.3])
    >>> result.classify()[:5]
    array([1, 1, 1, 1, 1])
    """
    from mirt.utils.data import response_column_names, validate_responses

    n_classes = _validate_count(n_classes, "n_classes", 1)
    max_iter = _validate_count(max_iter, "max_iter", 1)
    n_starts = _validate_count(n_starts, "n_starts", 1)
    if (
        isinstance(tol, (bool, np.bool_))
        or not isinstance(tol, (int, float, np.integer, np.floating))
        or not np.isfinite(tol)
        or tol <= 0.0
    ):
        raise MirtValidationError(
            "tol must be a positive finite number", parameter="tol", value=tol
        )
    if (
        isinstance(smoothing, (bool, np.bool_))
        or not isinstance(smoothing, (int, float, np.integer, np.floating))
        or not np.isfinite(smoothing)
        or smoothing < 0.0
    ):
        raise MirtValidationError(
            "smoothing must be a non-negative finite number",
            parameter="smoothing",
            value=smoothing,
        )
    if seed is not None and (not _is_integer(seed) or seed < 0):
        raise MirtValidationError(
            "seed must be a non-negative integer or None",
            parameter="seed",
            value=seed,
        )
    verbose = _validate_bool(verbose, "verbose")
    compute_standard_errors = _validate_bool(
        compute_standard_errors, "compute_standard_errors"
    )
    tol = float(tol)
    smoothing = float(smoothing)

    if item_names is None:
        item_names = response_column_names(data)
    values = validate_responses(data)
    n_items = values.shape[1]
    names = _validate_item_names(item_names, n_items)
    observed_items = values >= 0
    unobserved = np.flatnonzero(~np.any(observed_items, axis=0))
    if unobserved.size:
        raise MirtDataError(
            "every item needs at least one observed response; items "
            + ", ".join(names[index] for index in unobserved[:10])
            + " have none",
            n_items=n_items,
        )
    categories = _infer_categories(values, n_categories)
    offsets = np.concatenate([[0], np.cumsum(categories)]).astype(np.int_)
    item_of_column = np.repeat(np.arange(n_items), categories)

    # EM runs on the distinct response patterns weighted by their counts. A
    # pattern without observed responses gets weight zero; its posterior is
    # the class proportions.
    patterns, inverse, pattern_counts = np.unique(
        values, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    observed = (patterns >= 0).astype(np.float64)
    counts = pattern_counts * np.any(patterns >= 0, axis=1).astype(np.float64)
    indicators = _category_indicators(patterns, offsets)

    rng = np.random.default_rng(seed)
    best: _EMState | None = None
    start_log_likelihoods = np.empty(n_starts, dtype=np.float64)
    for start in range(n_starts):
        proportions, probabilities = _starting_values(
            n_classes, categories, None if start == 0 else rng
        )
        state = _run_em(
            indicators,
            counts,
            item_of_column,
            offsets,
            categories,
            proportions,
            probabilities,
            smoothing=smoothing,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            label=f"Start {start + 1}, " if n_starts > 1 else "",
        )
        start_log_likelihoods[start] = state.log_likelihood
        if best is None or state.objective > best.objective:
            best = state
    assert best is not None

    order = np.argsort(-best.class_proportions, kind="stable")
    class_proportions = best.class_proportions[order]
    probabilities = best.probabilities[order]
    posterior = best.posterior[:, order]

    model = LatentClassModel(
        class_proportions,
        [probabilities[:, offsets[j] : offsets[j + 1]] for j in range(n_items)],
        item_names=names,
    )
    proportion_se: NDArray[np.float64] | None = None
    probability_se: list[NDArray[np.float64]] | None = None
    if compute_standard_errors:
        flat_proportion_se, flat_probability_se = _standard_errors(
            indicators,
            observed,
            counts,
            posterior,
            class_proportions,
            probabilities,
            item_of_column,
            offsets,
            categories,
            smoothing,
        )
        proportion_se = flat_proportion_se
        probability_se = [
            flat_probability_se[:, offsets[j] : offsets[j + 1]].copy()
            for j in range(n_items)
        ]

    return LCAResult(
        model=model,
        log_likelihood=best.log_likelihood,
        n_parameters=model.n_parameters,
        n_observations=int(np.sum(counts)),
        n_iterations=best.n_iterations,
        converged=best.converged,
        class_posteriors=posterior[inverse],
        start_log_likelihoods=start_log_likelihoods,
        entropy=_relative_entropy(posterior, counts),
        class_proportions_se=proportion_se,
        item_probabilities_se=probability_se,
    )


def mdirt(
    data: Any,
    n_classes: int,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    *,
    n_starts: int = 1,
    seed: int | None = None,
    smoothing: float = 0.0,
    n_categories: int | Sequence[int] | None = None,
    item_names: Sequence[str] | None = None,
    compute_standard_errors: bool = True,
) -> LCAResult:
    """Fit a latent class model; an alias of :func:`fit_lca`.

    Named after R mirt's ``mdirt(data, model)``, whose integer ``model`` is
    ``n_classes`` here and whose ``nruns`` is ``n_starts``. See
    :func:`fit_lca` for the parameters.
    """
    return fit_lca(
        data,
        n_classes,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        n_starts=n_starts,
        seed=seed,
        smoothing=smoothing,
        n_categories=n_categories,
        item_names=item_names,
        compute_standard_errors=compute_standard_errors,
    )
