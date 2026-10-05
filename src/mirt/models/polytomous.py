from collections.abc import Callable, Iterator

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt._model_defaults import register_builtin_model as _register_builtin_model
from mirt._model_defaults import uses_builtin_model_hooks
from mirt.backends.rust._helpers import RUST_AVAILABLE
from mirt.backends.rust.polytomous import (
    compute_log_likelihoods_gpcm,
    compute_log_likelihoods_grm,
)
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtValidationError
from mirt.models.base import PolytomousItemModel

_MAX_PROBABILITY_CHUNK_ENTRIES = 1_000_000
_MAX_VECTORIZED_INFORMATION_ROWS = 256
_INFORMATION_HOOKS = ("_item_information", "probability", "_category_probabilities")


def _identify_rating_scale_origin(
    parameters: dict[str, NDArray[np.float64]],
) -> None:
    """Fix the first shared threshold while preserving every item boundary."""
    offset = float(parameters["thresholds"][0])
    parameters["difficulty"] += offset
    parameters["thresholds"] -= offset


def _category_count_chunks(
    category_counts: list[int],
    n_persons: int,
) -> Iterator[tuple[int, NDArray[np.intp]]]:
    """Group equal-width items into memory-bounded probability chunks."""
    counts = np.asarray(category_counts, dtype=np.intp)
    for n_categories in np.unique(counts):
        item_indices = np.flatnonzero(counts == n_categories)
        chunk_size = max(
            1,
            _MAX_PROBABILITY_CHUNK_ENTRIES // max(1, n_persons * int(n_categories)),
        )
        for start in range(0, item_indices.size, chunk_size):
            yield (
                int(n_categories),
                item_indices[start : start + chunk_size],
            )


def _stable_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute category-last softmax without overflowing exponentials."""
    weights = logits - np.max(logits, axis=-1, keepdims=True)
    np.exp(weights, out=weights)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights


def _graded_probabilities(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert cumulative logits to category-last graded probabilities."""
    cumulative = sigmoid(logits)
    probabilities = np.empty((*logits.shape[:-1], logits.shape[-1] + 1))
    probabilities[..., 0] = 1.0 - cumulative[..., 0]
    probabilities[..., 1:-1] = cumulative[..., :-1] - cumulative[..., 1:]
    probabilities[..., -1] = cumulative[..., -1]
    return probabilities


def _partial_credit_probabilities(
    increments: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert adjacent-category increments to stable probabilities."""
    logits = np.empty((*increments.shape[:-1], increments.shape[-1] + 1))
    logits[..., 0] = 0.0
    np.cumsum(increments, axis=-1, out=logits[..., 1:])
    return _stable_softmax(logits)


def _score_variance(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Retain small tail contributions when the expected score is saturated.

    Categories are on the last axis; zero-probability padding adds nothing.
    """
    categories = np.arange(probabilities.shape[-1])
    mean = probabilities @ categories
    return np.sum(probabilities * (categories - mean[..., None]) ** 2, axis=-1)


def _graded_information(
    probabilities: NDArray[np.float64],
    discrimination: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute graded-response information from existing category curves.

    Categories are on the last axis. ``discrimination`` broadcasts against
    the leading axes after a trailing category axis is appended, so a
    ``(n_items,)`` slope vector scales ``(n_persons, n_items, width)`` curves
    whose zero-probability padding adds nothing.
    """
    cumulative = np.empty(
        (*probabilities.shape[:-1], probabilities.shape[-1] + 1), dtype=np.float64
    )
    cumulative[..., 0] = 1.0
    cumulative[..., -1] = 0.0
    np.cumsum(
        probabilities[..., :0:-1],
        axis=-1,
        out=cumulative[..., -2:0:-1],
    )

    np.multiply(cumulative, 1.0 - cumulative, out=cumulative)
    derivatives = cumulative[..., :-1] - cumulative[..., 1:]
    derivatives *= np.asarray(discrimination)[..., None]
    np.square(derivatives, out=derivatives)
    valid = probabilities > PROB_EPSILON
    np.divide(
        derivatives,
        probabilities,
        out=derivatives,
        where=valid,
    )
    derivatives[~valid] = 0.0
    return derivatives.sum(axis=-1)


def _uses_vectorized_information(
    model: PolytomousItemModel, owner: type[PolytomousItemModel]
) -> bool:
    """Whether ``owner``'s all-item information kernel describes ``model``.

    Instance or class replacements of ``_item_information`` or the curve
    hooks keep the per-item path. The rating-scale families are not
    registered built-ins, so only their exact classes qualify.
    """
    if not vars(model).keys().isdisjoint(_INFORMATION_HOOKS):
        return False
    model_class = type(model)
    authored = _AUTHORED_INFORMATION_HOOKS[owner]
    if any(getattr(model_class, name) is not hook for name, hook in authored.items()):
        return False
    return uses_builtin_model_hooks(model) or (
        model_class is owner and owner in (RatingScaleModel, GradedRatingScaleModel)
    )


def _vectorized_item_information(
    model: PolytomousItemModel,
    owner: type[PolytomousItemModel],
    theta: NDArray[np.float64],
    kernel: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Evaluate all items at once for small ability batches.

    The all-item kernel removes per-item overhead, which dominates adaptive
    testing and short grids. Larger batches keep the per-item path, whose
    curves stay cache resident.
    """
    theta = model._ensure_theta_2d(theta)
    if theta.shape[0] > _MAX_VECTORIZED_INFORMATION_ROWS or (
        not _uses_vectorized_information(model, owner)
    ):
        return PolytomousItemModel._information_by_item(model, theta)
    information = np.empty((theta.shape[0], model.n_items), dtype=np.float64)
    width = model.n_items * model.max_categories * model.n_factors
    rows_per_block = max(1, _MAX_PROBABILITY_CHUNK_ENTRIES // width)
    for start in range(0, theta.shape[0], rows_per_block):
        stop = start + rows_per_block
        information[start:stop] = kernel(theta[start:stop])
    return information


def _sum_item_information_matrices(
    model: "GradedResponseModel | GeneralizedPartialCredit | NominalResponseModel",
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum item Fisher matrices across conditionally independent items."""
    theta = model._ensure_theta_2d(theta)
    information = np.zeros((len(theta), model.n_factors, model.n_factors))
    for item_idx in range(model.n_items):
        information += model.item_information_matrix(theta, item_idx)
    return information


@_register_builtin_model
class GradedResponseModel(PolytomousItemModel):
    model_name = "GRM"
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        if self.n_factors == 1:
            self._parameters["discrimination"] = np.ones(self.n_items)
        else:
            self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))

        max_cats = max(self._n_categories)
        thresholds = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                thresholds[i, : n_cat - 1] = np.linspace(-2, 2, n_cat - 1)

        self._parameters["thresholds"] = thresholds

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        return self._parameters["thresholds"]

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        masks["thresholds"] = self._category_columns(self.thresholds.shape[1], 1)
        return self._apply_free_parameter_restrictions(masks)

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name == "thresholds":
            canonical[~self._category_columns(canonical.shape[1], 1)] = 0.0
        return canonical

    def cumulative_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        threshold_idx: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)

        a = self._parameters["discrimination"]
        b = self._parameters["thresholds"][item_idx, threshold_idx]

        if self.n_factors == 1:
            a_item = a[item_idx]
            z = a_item * (theta.ravel() - b)
        else:
            a_item = a[item_idx]
            z = np.dot(theta, a_item) - np.sum(a_item) * b

        return sigmoid(z)

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        return self._category_probabilities(theta, item_idx)[:, category]

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all GRM category probabilities in one vectorized pass."""
        theta = self._ensure_theta_2d(theta)
        n_cat = self._n_categories[item_idx]
        a_item = self._parameters["discrimination"][item_idx]
        thresholds = self._parameters["thresholds"][item_idx, : n_cat - 1]

        if self.n_factors == 1:
            logits = a_item * (theta.ravel()[:, None] - thresholds[None, :])
        else:
            logits = np.dot(theta, a_item)[:, None] - np.sum(a_item) * thresholds

        return _graded_probabilities(logits)

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute all-item GRM probabilities in bounded item chunks."""
        theta = self._ensure_theta_2d(theta)
        if item_idx is not None:
            return self._category_probabilities(theta, item_idx)
        if self.n_items == 1:
            return self._category_probabilities(theta, 0)[:, None, :]
        n_persons = theta.shape[0]
        probabilities = np.zeros(
            (n_persons, self.n_items, max(self._n_categories)),
            dtype=np.float64,
        )
        discrimination = self._parameters["discrimination"]
        thresholds = self._parameters["thresholds"]

        for n_categories, item_indices in _category_count_chunks(
            self._n_categories, n_persons
        ):
            active_thresholds = thresholds[item_indices, : n_categories - 1]
            active_discrimination = discrimination[item_indices]
            if self.n_factors == 1:
                logits = active_discrimination[None, :, None] * (
                    theta[:, 0, None, None] - active_thresholds[None, :, :]
                )
            else:
                projected_theta = theta @ active_discrimination.T
                threshold_scale = np.sum(active_discrimination, axis=1)
                logits = (
                    projected_theta[:, :, None]
                    - (threshold_scale[:, None] * active_thresholds)[None, :, :]
                )

            probabilities[:, item_indices, :n_categories] = _graded_probabilities(
                logits
            )

        return probabilities

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item GRM category probabilities."""
        theta_2d, indices = self._prepare_probability_pairs(theta, item_indices)
        category_counts = np.asarray(self._n_categories, dtype=np.intp)[indices]
        result = np.zeros((indices.size, max(self._n_categories)), dtype=np.float64)
        discrimination = self._parameters["discrimination"][indices]
        thresholds = self._parameters["thresholds"][indices]

        for n_categories in np.unique(category_counts):
            selected = category_counts == n_categories
            active_thresholds = thresholds[selected, : n_categories - 1]
            active_discrimination = discrimination[selected]
            if self.n_factors == 1:
                logits = active_discrimination[:, None] * (
                    theta_2d[selected, 0, None] - active_thresholds
                )
            else:
                projected = np.einsum(
                    "ij,ij->i",
                    theta_2d[selected],
                    active_discrimination,
                )
                threshold_scale = np.sum(active_discrimination, axis=1)
                logits = (
                    projected[:, None] - threshold_scale[:, None] * active_thresholds
                )
            result[selected, :n_categories] = _graded_probabilities(logits)

        return result

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        a = self._parameters["discrimination"]
        if self.n_factors == 1:
            a_val = float(a[item_idx])
        else:
            a_val = float(np.linalg.norm(a[item_idx]))

        probabilities = self._category_probabilities(theta, item_idx)
        return _graded_information(probabilities, a_val)

    def _information_by_item(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        discrimination = self._parameters["discrimination"]
        if self.n_factors > 1:
            discrimination = np.linalg.norm(discrimination, axis=1)
        return _vectorized_item_information(
            self,
            GradedResponseModel,
            theta,
            lambda block: _graded_information(self.probability(block), discrimination),
        )

    def item_information_matrix(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Return exact Fisher matrices of shape ``(n_persons, n_factors, n_factors)``.

        Every cumulative logit has the slope vector as its ability gradient,
        so each matrix is the unit-slope graded information times the outer
        product of that vector. The scalar ``information`` method returns the
        trace of this matrix.
        """
        item_idx = self._validate_item_index(item_idx)
        theta = self._ensure_theta_2d(theta)
        slope = np.asarray(self._parameters["discrimination"][item_idx]).reshape(-1)
        information = _graded_information(self.probability(theta, item_idx), 1.0)
        return information[:, None, None] * np.outer(slope, slope)

    def test_information_matrix(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sum item Fisher matrices across conditionally independent items."""
        return _sum_item_information_matrices(self, theta)

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if RUST_AVAILABLE and self.n_factors == 1 and uses_builtin_model_hooks(self):
            responses = self._validate_polytomous_responses(responses)
            theta = self._ensure_theta_2d(theta)
            quad_points = theta.ravel() if theta.ndim == 2 else theta
            disc = self._parameters["discrimination"]
            thresh = self._parameters["thresholds"]
            n_cats = np.array(self._n_categories, dtype=np.int32)
            return compute_log_likelihoods_grm(
                responses, quad_points, disc, thresh, n_cats
            )
        return super().log_likelihood_batch(responses, theta)


@_register_builtin_model
class GeneralizedPartialCredit(PolytomousItemModel):
    """Generalized partial credit model with centered step thresholds.

    Adjacent categories have log odds ``a * (theta - step)`` in one
    dimension and ``a @ theta - sum(a) * step`` in multiple dimensions.
    Each factor loading therefore remains the slope with respect to that
    factor. A two-category item reduces to the centered-threshold 2PL, and
    adding factors with zero loadings preserves the original response curve.
    """

    model_name = "GPCM"
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        if self.n_factors == 1:
            self._parameters["discrimination"] = np.ones(self.n_items)
        else:
            self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))

        max_cats = max(self._n_categories)
        steps = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                steps[i, : n_cat - 1] = np.linspace(-1, 1, n_cat - 1)

        self._parameters["steps"] = steps

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def steps(self) -> NDArray[np.float64]:
        return self._parameters["steps"]

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        masks["steps"] = self._category_columns(self.steps.shape[1], 1)
        return self._apply_free_parameter_restrictions(masks)

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name == "steps":
            canonical[~self._category_columns(canonical.shape[1], 1)] = 0.0
        return canonical

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        return self._category_probabilities(theta, item_idx)[:, category]

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all GPCM category probabilities in one stable pass."""
        theta = self._ensure_theta_2d(theta)
        n_cat = self._n_categories[item_idx]

        a = self._parameters["discrimination"]
        steps = self._parameters["steps"][item_idx, : n_cat - 1]

        if self.n_factors == 1:
            a_item = a[item_idx]
            increments = a_item * (theta.ravel()[:, None] - steps[None, :])
        else:
            a_item = a[item_idx]
            projected_theta = np.dot(theta, a_item)
            increments = projected_theta[:, None] - np.sum(a_item) * steps[None, :]

        return _partial_credit_probabilities(increments)

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute all-item GPCM probabilities in bounded item chunks."""
        theta = self._ensure_theta_2d(theta)
        if item_idx is not None:
            return self._category_probabilities(theta, item_idx)
        if self.n_items == 1:
            return self._category_probabilities(theta, 0)[:, None, :]

        n_persons = theta.shape[0]
        probabilities = np.zeros(
            (n_persons, self.n_items, max(self._n_categories)),
            dtype=np.float64,
        )
        discrimination = self._parameters["discrimination"]
        steps = self._parameters["steps"]

        for n_categories, item_indices in _category_count_chunks(
            self._n_categories, n_persons
        ):
            active_steps = steps[item_indices, : n_categories - 1]
            active_discrimination = discrimination[item_indices]
            if self.n_factors == 1:
                increments = active_discrimination[None, :, None] * (
                    theta[:, 0, None, None] - active_steps[None, :, :]
                )
            else:
                projected_theta = theta @ active_discrimination.T
                threshold_scale = np.sum(active_discrimination, axis=1)
                increments = (
                    projected_theta[:, :, None]
                    - (threshold_scale[:, None] * active_steps)[None, :, :]
                )

            probabilities[:, item_indices, :n_categories] = (
                _partial_credit_probabilities(increments)
            )

        return probabilities

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item GPCM category probabilities."""
        theta_2d, indices = self._prepare_probability_pairs(theta, item_indices)
        category_counts = np.asarray(self._n_categories, dtype=np.intp)[indices]
        result = np.zeros((indices.size, max(self._n_categories)), dtype=np.float64)
        discrimination = self._parameters["discrimination"][indices]
        steps = self._parameters["steps"][indices]

        for n_categories in np.unique(category_counts):
            selected = category_counts == n_categories
            active_steps = steps[selected, : n_categories - 1]
            active_discrimination = discrimination[selected]
            if self.n_factors == 1:
                increments = active_discrimination[:, None] * (
                    theta_2d[selected, 0, None] - active_steps
                )
            else:
                projected = np.einsum(
                    "ij,ij->i",
                    theta_2d[selected],
                    active_discrimination,
                )
                threshold_scale = np.sum(active_discrimination, axis=1)
                increments = (
                    projected[:, None] - threshold_scale[:, None] * active_steps
                )
            result[selected, :n_categories] = _partial_credit_probabilities(increments)

        return result

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        a = self._parameters["discrimination"]
        if self.n_factors == 1:
            slope_squared = a[item_idx] ** 2
        else:
            slope_squared = np.dot(a[item_idx], a[item_idx])
        return slope_squared * _score_variance(self.probability(theta, item_idx))

    def _information_by_item(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        a = self._parameters["discrimination"]
        slope_squared = a**2 if self.n_factors == 1 else np.einsum("jf,jf->j", a, a)
        return _vectorized_item_information(
            self,
            GeneralizedPartialCredit,
            theta,
            lambda block: slope_squared * _score_variance(self.probability(block)),
        )

    def item_information_matrix(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Return exact Fisher matrices of shape ``(n_persons, n_factors, n_factors)``.

        The adjacent logits share the slope vector, so each matrix is the
        conditional score variance times the outer product of that vector.
        The scalar ``information`` method returns the trace of this matrix.
        """
        item_idx = self._validate_item_index(item_idx)
        theta = self._ensure_theta_2d(theta)
        slope = np.asarray(self._parameters["discrimination"][item_idx]).reshape(-1)
        variance = _score_variance(self.probability(theta, item_idx))
        return variance[:, None, None] * np.outer(slope, slope)

    def test_information_matrix(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sum item Fisher matrices across conditionally independent items."""
        return _sum_item_information_matrices(self, theta)

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if RUST_AVAILABLE and self.n_factors == 1 and uses_builtin_model_hooks(self):
            responses = self._validate_polytomous_responses(responses)
            theta = self._ensure_theta_2d(theta)
            quad_points = theta.ravel() if theta.ndim == 2 else theta
            disc = self._parameters["discrimination"]
            steps_full = np.zeros((self.n_items, max(self._n_categories)))
            for i, n_cat in enumerate(self._n_categories):
                steps_full[i, 1:n_cat] = self._parameters["steps"][i, : n_cat - 1]
            n_cats = np.array(self._n_categories, dtype=np.int32)
            return compute_log_likelihoods_gpcm(
                responses, quad_points, disc, steps_full, n_cats
            )
        return super().log_likelihood_batch(responses, theta)


@_register_builtin_model
class PartialCreditModel(GeneralizedPartialCredit):
    model_name = "PCM"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("PCM only supports unidimensional analysis")
        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items)

        max_cats = max(self._n_categories)
        steps = np.zeros((self.n_items, max_cats - 1))

        for i, n_cat in enumerate(self._n_categories):
            if n_cat > 1:
                steps[i, : n_cat - 1] = np.linspace(-1, 1, n_cat - 1)

        self._parameters["steps"] = steps

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        masks["discrimination"] = np.zeros_like(self.discrimination, dtype=np.bool_)
        return self._apply_free_parameter_restrictions(masks)

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name == "discrimination":
            canonical.fill(1.0)
        return canonical

    def set_parameters(self, **params: NDArray[np.float64]) -> "PartialCreditModel":
        if "discrimination" in params:
            raise ValueError("Cannot set discrimination in PCM (fixed to 1)")
        return super().set_parameters(**params)


class RatingScaleModel(PolytomousItemModel):
    """Rating Scale Model (RSM) for polytomous items.

    The RSM is a special case of the Partial Credit Model where step
    parameters are constrained to be equal across all items. This is
    appropriate when all items share the same rating scale structure
    (e.g., Likert scales with the same response options).

    Parameters
    ----------
    n_items : int
        Number of items
    n_categories : int
        Number of response categories (must be same for all items)
    item_names : list of str, optional
        Names for each item

    Attributes
    ----------
    difficulty : ndarray of shape (n_items,)
        Item location/difficulty parameters
    thresholds : ndarray of shape (n_categories - 1,)
        Step thresholds shared across all items

    Notes
    -----
    The probability of responding in category k for item j is:

        P(X_j = k | theta) = exp(sum_{v=0}^{k} (theta - b_j - tau_v)) /
                             sum_{c=0}^{K} exp(sum_{v=0}^{c} (theta - b_j - tau_v))

    where b_j is the item difficulty and tau_v are the shared thresholds.

    The RSM reduces the number of parameters compared to GPCM/PCM,
    which can be beneficial when the assumption of equal thresholds
    is reasonable.

    References
    ----------
    Andrich, D. (1978). A rating formulation for ordered response categories.
        Psychometrika, 43(4), 561-573.
    """

    model_name = "RSM"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("RSM only supports unidimensional analysis")

        if isinstance(n_categories, list):
            if len(set(n_categories)) != 1:
                raise ValueError(
                    "RSM requires all items to have the same number of categories"
                )
            n_categories = n_categories[0]

        self._n_cats = n_categories
        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["difficulty"] = np.zeros(self.n_items)

        n_thresholds = self._n_cats - 1
        self._parameters["thresholds"] = np.linspace(-1, 1, n_thresholds)
        _identify_rating_scale_origin(self._parameters)

    @property
    def difficulty(self) -> NDArray[np.float64]:
        """Item difficulty/location parameters."""
        return self._parameters["difficulty"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        """Shared step threshold parameters."""
        return self._parameters["thresholds"]

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        masks["thresholds"][0] = False
        return self._apply_free_parameter_restrictions(masks)

    def _canonical_parameter_values(
        self, name: str, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name == "thresholds":
            canonical -= canonical[0]
        return canonical

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute probability of responding in a specific category.

        Parameters
        ----------
        theta : ndarray
            Ability values
        item_idx : int
            Item index
        category : int
            Response category (0 to n_categories - 1)

        Returns
        -------
        ndarray
            Probability of category response for each theta
        """
        theta = self._ensure_theta_2d(theta)
        n_cat = self._n_cats

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        return self._category_probabilities(theta, item_idx)[:, category]

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all RSM category probabilities in one stable pass."""
        theta = self._ensure_theta_2d(theta)

        b_j = self._parameters["difficulty"][item_idx]
        tau = self._parameters["thresholds"]
        increments = theta.ravel()[:, None] - b_j - tau[None, :]

        return _partial_credit_probabilities(increments)

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute all-item RSM probabilities in bounded item chunks."""
        theta = self._ensure_theta_2d(theta)
        if item_idx is not None:
            return self._category_probabilities(theta, item_idx)
        if self.n_items == 1:
            return self._category_probabilities(theta, 0)[:, None, :]

        n_persons = theta.shape[0]
        probabilities = np.empty(
            (n_persons, self.n_items, self._n_cats),
            dtype=np.float64,
        )
        difficulty = self._parameters["difficulty"]
        thresholds = self._parameters["thresholds"]
        for _, item_indices in _category_count_chunks(self._n_categories, n_persons):
            increments = (
                theta[:, 0, None, None]
                - difficulty[item_indices][None, :, None]
                - thresholds[None, None, :]
            )
            probabilities[:, item_indices, :] = _partial_credit_probabilities(
                increments
            )
        return probabilities

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item RSM category probabilities."""
        theta_2d, indices = self._prepare_probability_pairs(theta, item_indices)
        increments = (
            theta_2d[:, 0, None]
            - self._parameters["difficulty"][indices, None]
            - self._parameters["thresholds"][None, :]
        )
        return _partial_credit_probabilities(increments)

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute item information function.

        Uses the variance of the item score as the information.
        """
        return _score_variance(self.probability(theta, item_idx))

    def _information_by_item(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return _vectorized_item_information(
            self,
            RatingScaleModel,
            theta,
            lambda block: _score_variance(self.probability(block)),
        )

    def set_parameters(self, **params: NDArray[np.float64]) -> "RatingScaleModel":
        """Set model parameters, preserving curves with the first threshold zero.

        Parameters
        ----------
        difficulty : ndarray of shape (n_items,)
            Item difficulty parameters
        thresholds : ndarray of shape (n_categories - 1,)
            Shared threshold parameters

        Returns
        -------
        self
        """
        candidates = {name: values.copy() for name, values in self._parameters.items()}
        for name, values in params.items():
            if name not in candidates:
                raise ValueError(f"Unknown parameter: {name}")
            values = np.array(values, dtype=np.float64, copy=True)
            if name == "difficulty" and values.shape != (self.n_items,):
                raise ValueError(f"difficulty must have shape ({self.n_items},)")
            if name == "thresholds" and values.shape != (self._n_cats - 1,):
                raise ValueError(f"thresholds must have shape ({self._n_cats - 1},)")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain finite values")
            candidates[name] = values

        _identify_rating_scale_origin(candidates)
        if not all(np.all(np.isfinite(values)) for values in candidates.values()):
            raise ValueError("identified rating-scale parameters must be finite")
        self._parameters = candidates

        self._is_fitted = True
        return self

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        if param_name == "thresholds":
            raise MirtValidationError("thresholds are shared; use set_parameters")
        super().set_item_parameter(item_idx, param_name, value)


class GradedRatingScaleModel(PolytomousItemModel):
    """Graded Rating Scale Model (GRSM) for polytomous items.

    The GRSM is a constrained GRM where discrimination parameters are
    equal across all items. This is the graded response analog of the
    Rating Scale Model.

    Parameters
    ----------
    n_items : int
        Number of items
    n_categories : int
        Number of response categories (must be same for all items)
    item_names : list of str, optional
        Names for each item

    Attributes
    ----------
    discrimination : float
        Common discrimination parameter for all items
    difficulty : ndarray of shape (n_items,)
        Item location parameters
    thresholds : ndarray of shape (n_categories - 1,)
        Category threshold parameters (relative to item location)

    Notes
    -----
    The GRSM cumulative probability is:

        P(X >= k | theta) = 1 / (1 + exp(-a * (theta - b_j - tau_k)))

    where a is the common discrimination, b_j is item location, and
    tau_k are shared category thresholds.

    References
    ----------
    Muraki, E. (1990). Fitting a polytomous item response model to
        Likert-type data. Applied Psychological Measurement, 14, 59-71.
    """

    model_name = "GRSM"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("GRSM only supports unidimensional analysis")

        if isinstance(n_categories, list):
            if len(set(n_categories)) != 1:
                raise ValueError(
                    "GRSM requires all items to have the same number of categories"
                )
            n_categories = n_categories[0]

        self._n_cats = n_categories
        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.array([1.0])
        self._parameters["difficulty"] = np.zeros(self.n_items)
        n_thresholds = self._n_cats - 1
        self._parameters["thresholds"] = np.linspace(-2, 2, n_thresholds)
        _identify_rating_scale_origin(self._parameters)

    @property
    def discrimination(self) -> float:
        """Common discrimination parameter."""
        return float(self._parameters["discrimination"][0])

    @property
    def difficulty(self) -> NDArray[np.float64]:
        """Item location parameters."""
        return self._parameters["difficulty"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        """Shared category threshold parameters."""
        return self._parameters["thresholds"]

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        masks["thresholds"][0] = False
        return self._apply_free_parameter_restrictions(masks)

    def _canonical_parameter_values(
        self, name: str, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name == "thresholds":
            canonical -= canonical[0]
        return canonical

    def cumulative_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        threshold_idx: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"][0]
        b_j = self._parameters["difficulty"][item_idx]
        tau_k = self._parameters["thresholds"][threshold_idx]

        z = a * (theta_1d - b_j - tau_k)
        return sigmoid(z)

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        n_cat = self._n_cats

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        return self._category_probabilities(theta, item_idx)[:, category]

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all GRSM category probabilities in one vectorized pass."""
        theta = self._ensure_theta_2d(theta)
        a = self._parameters["discrimination"][0]
        b_j = self._parameters["difficulty"][item_idx]
        thresholds = self._parameters["thresholds"]

        logits = a * (theta.ravel()[:, None] - b_j - thresholds[None, :])
        return _graded_probabilities(logits)

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute all-item GRSM probabilities in bounded item chunks."""
        theta = self._ensure_theta_2d(theta)
        if item_idx is not None:
            return self._category_probabilities(theta, item_idx)
        if self.n_items == 1:
            return self._category_probabilities(theta, 0)[:, None, :]

        n_persons = theta.shape[0]
        probabilities = np.empty(
            (n_persons, self.n_items, self._n_cats),
            dtype=np.float64,
        )
        discrimination = self._parameters["discrimination"][0]
        difficulty = self._parameters["difficulty"]
        thresholds = self._parameters["thresholds"]
        for _, item_indices in _category_count_chunks(self._n_categories, n_persons):
            logits = discrimination * (
                theta[:, 0, None, None]
                - difficulty[item_indices][None, :, None]
                - thresholds[None, None, :]
            )
            probabilities[:, item_indices, :] = _graded_probabilities(logits)
        return probabilities

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item GRSM category probabilities."""
        theta_2d, indices = self._prepare_probability_pairs(theta, item_indices)
        logits = self._parameters["discrimination"][0] * (
            theta_2d[:, 0, None]
            - self._parameters["difficulty"][indices, None]
            - self._parameters["thresholds"][None, :]
        )
        return _graded_probabilities(logits)

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        discrimination = float(self._parameters["discrimination"][0])
        probabilities = self._category_probabilities(theta, item_idx)
        return _graded_information(probabilities, discrimination)

    def _information_by_item(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        discrimination = float(self._parameters["discrimination"][0])
        return _vectorized_item_information(
            self,
            GradedRatingScaleModel,
            theta,
            lambda block: _graded_information(self.probability(block), discrimination),
        )

    def set_parameters(
        self,
        discrimination: float | None = None,
        difficulty: NDArray[np.float64] | None = None,
        thresholds: NDArray[np.float64] | None = None,
    ) -> "GradedRatingScaleModel":
        """Set parameters atomically, absorbing the first threshold into location."""
        candidates = {name: values.copy() for name, values in self._parameters.items()}
        if discrimination is not None:
            values = np.asarray(discrimination, dtype=np.float64)
            if values.size != 1 or values.ndim > 1:
                raise ValueError("discrimination must be a positive scalar")
            scalar = float(values.item())
            if not np.isfinite(scalar) or scalar <= 0.0:
                raise ValueError("discrimination must be a finite positive scalar")
            candidates["discrimination"] = np.array([scalar])
        if difficulty is not None:
            difficulty = np.array(difficulty, dtype=np.float64, copy=True)
            if difficulty.shape != (self.n_items,):
                raise ValueError(f"difficulty must have shape ({self.n_items},)")
            if not np.all(np.isfinite(difficulty)):
                raise ValueError("difficulty must contain finite values")
            candidates["difficulty"] = difficulty
        if thresholds is not None:
            thresholds = np.array(thresholds, dtype=np.float64, copy=True)
            if thresholds.shape != (self._n_cats - 1,):
                raise ValueError(f"thresholds must have shape ({self._n_cats - 1},)")
            if not np.all(np.isfinite(thresholds)) or np.any(np.diff(thresholds) < 0.0):
                raise ValueError("thresholds must be finite and nondecreasing")
            candidates["thresholds"] = thresholds

        _identify_rating_scale_origin(candidates)
        if not all(np.all(np.isfinite(values)) for values in candidates.values()):
            raise ValueError("identified rating-scale parameters must be finite")
        self._parameters = candidates

        self._is_fitted = True
        return self

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        if param_name in {"thresholds", "discrimination"}:
            raise MirtValidationError(f"{param_name} is shared; use set_parameters")
        super().set_item_parameter(item_idx, param_name, value)


@_register_builtin_model
class NominalResponseModel(PolytomousItemModel):
    model_name = "NRM"
    supports_multidimensional = True

    def _initialize_parameters(self) -> None:
        max_cats = max(self._n_categories)

        if self.n_factors == 1:
            slopes = np.zeros((self.n_items, max_cats))
            for i, n_cat in enumerate(self._n_categories):
                slopes[i, 1:n_cat] = np.linspace(0.5, 1.5, n_cat - 1)
        else:
            slopes = np.zeros((self.n_items, max_cats, self.n_factors))
            for i, n_cat in enumerate(self._n_categories):
                for f in range(self.n_factors):
                    slopes[i, 1:n_cat, f] = np.linspace(0.5, 1.5, n_cat - 1)

        self._parameters["slopes"] = slopes

        intercepts = np.zeros((self.n_items, max_cats))
        for i, n_cat in enumerate(self._n_categories):
            intercepts[i, 1:n_cat] = np.linspace(-1, 1, n_cat - 1)

        self._parameters["intercepts"] = intercepts

    @property
    def slopes(self) -> NDArray[np.float64]:
        return self._parameters["slopes"]

    @property
    def intercepts(self) -> NDArray[np.float64]:
        return self._parameters["intercepts"]

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        intercept_mask = self._category_columns(self.intercepts.shape[1])
        intercept_mask[:, 0] = False
        slope_mask = np.zeros_like(self.slopes, dtype=np.bool_)
        slope_mask[intercept_mask] = True
        masks["slopes"] = slope_mask
        masks["intercepts"] = intercept_mask
        return self._apply_free_parameter_restrictions(masks)

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name not in {"slopes", "intercepts"}:
            return canonical

        canonical -= canonical[:, :1].copy()
        canonical[~self._category_columns(canonical.shape[1])] = 0.0
        return canonical

    def set_parameters(self, **params: NDArray[np.float64]) -> "NominalResponseModel":
        """Store probability-equivalent category contrasts with reference zero."""
        converted = {}
        for name, values in params.items():
            if name not in self._parameters:
                raise MirtValidationError(f"Unknown parameter: {name}", parameter=name)
            array = np.array(values, dtype=np.float64, copy=True)
            if array.shape != self._parameters[name].shape:
                raise MirtValidationError(
                    f"Shape mismatch for {name}: expected {self._parameters[name].shape}",
                    parameter=name,
                )
            if not np.all(np.isfinite(array)):
                raise MirtValidationError(
                    f"{name} must contain finite values", parameter=name
                )
            canonical = self._canonical_parameter_values(name, array)
            if not np.all(np.isfinite(canonical)):
                raise MirtValidationError(
                    f"Identified {name} must contain finite values", parameter=name
                )
            converted[name] = canonical
        return super().set_parameters(**converted)

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_cat = self._n_categories[item_idx]

        if category < 0 or category >= n_cat:
            raise ValueError(f"Category {category} out of range [0, {n_cat})")

        return self._category_probabilities(theta, item_idx)[:, category]

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all NRM category probabilities in one stable pass."""
        theta = self._ensure_theta_2d(theta)
        n_cat = self._n_categories[item_idx]

        a = self._parameters["slopes"]
        c = self._parameters["intercepts"]

        if self.n_factors == 1:
            logits = (
                theta.ravel()[:, None] * a[item_idx, None, :n_cat]
                + c[item_idx, None, :n_cat]
            )
        else:
            logits = np.dot(theta, a[item_idx, :n_cat].T) + c[item_idx, None, :n_cat]

        return _stable_softmax(logits)

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute all-item NRM probabilities in bounded item chunks."""
        theta = self._ensure_theta_2d(theta)
        if item_idx is not None:
            return self._category_probabilities(theta, item_idx)
        if self.n_items == 1:
            return self._category_probabilities(theta, 0)[:, None, :]
        if self.n_items < 8:
            return super().probability(theta)

        n_persons = theta.shape[0]
        probabilities = np.zeros(
            (n_persons, self.n_items, max(self._n_categories)),
            dtype=np.float64,
        )
        slopes = self._parameters["slopes"]
        intercepts = self._parameters["intercepts"]

        for n_categories, item_indices in _category_count_chunks(
            self._n_categories, n_persons
        ):
            if item_indices.size < 4:
                for item_index in item_indices:
                    probabilities[:, item_index, :n_categories] = (
                        self._category_probabilities(theta, int(item_index))
                    )
                continue

            active_slopes = slopes[item_indices, :n_categories]
            active_intercepts = intercepts[item_indices, :n_categories]
            if self.n_factors == 1:
                logits = (
                    theta[:, 0, None, None] * active_slopes[None, :, :]
                    + active_intercepts[None, :, :]
                )
            else:
                logits = np.einsum(
                    "pf,icf->pic",
                    theta,
                    active_slopes,
                    optimize=True,
                )
                logits += active_intercepts[None, :, :]
            probabilities[:, item_indices, :n_categories] = _stable_softmax(logits)

        return probabilities

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item NRM category probabilities."""
        theta_2d, indices = self._prepare_probability_pairs(theta, item_indices)
        slopes = self._parameters["slopes"][indices]
        intercepts = self._parameters["intercepts"][indices]
        if self.n_factors == 1:
            logits = theta_2d[:, 0, None] * slopes + intercepts
        else:
            logits = np.einsum(
                "pf,pcf->pc",
                theta_2d,
                slopes,
                optimize=True,
            )
            logits += intercepts

        category_counts = np.asarray(self._n_categories, dtype=np.intp)[indices]
        categories = np.arange(self.max_categories)[None, :]
        logits = np.where(categories < category_counts[:, None], logits, -np.inf)
        return _stable_softmax(logits)

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        a = self._parameters["slopes"]
        probs = self.probability(theta, item_idx)

        if self.n_factors == 1:
            a_item = a[item_idx, :n_cat]

            expected_a = np.sum(probs * a_item, axis=1)

            expected_a_sq = np.sum(probs * (a_item**2), axis=1)

            info = expected_a_sq - expected_a**2
        else:
            info = np.zeros(n_persons)
            for f in range(self.n_factors):
                a_f = a[item_idx, :n_cat, f]
                expected_a = np.sum(probs * a_f, axis=1)
                expected_a_sq = np.sum(probs * (a_f**2), axis=1)
                info += expected_a_sq - expected_a**2

        return info

    def _information_by_item(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        slopes = self._parameters["slopes"].reshape(
            self.n_items, self.max_categories, self.n_factors
        )
        squared_slopes = slopes**2

        def kernel(block: NDArray[np.float64]) -> NDArray[np.float64]:
            probabilities = self.probability(block)
            expected = np.einsum("njc,jcf->njf", probabilities, slopes)
            expected_squared = np.einsum("njc,jcf->njf", probabilities, squared_slopes)
            return np.sum(expected_squared - expected**2, axis=2)

        return _vectorized_item_information(self, NominalResponseModel, theta, kernel)

    def item_information_matrix(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Return exact Fisher matrices of shape ``(n_persons, n_factors, n_factors)``.

        Category ``c`` has the slope vector ``a_c`` as its logit gradient, so
        each matrix is the covariance of those vectors under the category
        probabilities. The scalar ``information`` method returns the trace of
        this matrix.
        """
        item_idx = self._validate_item_index(item_idx)
        theta = self._ensure_theta_2d(theta)
        n_categories = self._n_categories[item_idx]
        slopes = self._parameters["slopes"][item_idx, :n_categories].reshape(
            n_categories, self.n_factors
        )
        probabilities = self.probability(theta, item_idx)
        centered = slopes[None, :, :] - (probabilities @ slopes)[:, None, :]
        return np.einsum("nc,ncf,ncg->nfg", probabilities, centered, centered)

    def test_information_matrix(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sum item Fisher matrices across conditionally independent items."""
        return _sum_item_information_matrices(self, theta)


# Captured at import so later class-level replacements are detected.
_AUTHORED_INFORMATION_HOOKS: dict[type[PolytomousItemModel], dict[str, object]] = {
    owner: {name: getattr(owner, name) for name in _INFORMATION_HOOKS}
    for owner in (
        GradedResponseModel,
        GeneralizedPartialCredit,
        RatingScaleModel,
        GradedRatingScaleModel,
        NominalResponseModel,
    )
}
