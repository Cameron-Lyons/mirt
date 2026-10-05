"""Item selection strategies for computerized adaptive testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.cat._native import register_native_defaults as _register_native_defaults
from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

# Bound (ability points x items) information blocks for large item pools.
_MAX_INFORMATION_VALUES = 131_072


def _validate_candidate_items(
    model: BaseItemModel,
    available_items: set[int],
) -> list[int]:
    """Return sorted candidate indices, rejecting invalid model access."""
    normalized: list[int] = []
    for item_idx in available_items:
        if isinstance(item_idx, (bool, np.bool_)) or not isinstance(item_idx, Integral):
            raise ValueError("available item indices must be integers")
        item = int(item_idx)
        if item < 0 or item >= model.n_items:
            raise ValueError(
                f"available item {item} is out of range [0, {model.n_items})"
            )
        normalized.append(item)
    normalized.sort()
    return normalized


def _information_at(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_idx: int,
) -> NDArray[np.float64]:
    """Return one item's information at each row of a 2D ability array."""
    information = np.asarray(model.information(theta, item_idx=item_idx), dtype=float)
    return information.reshape(len(theta), -1).sum(axis=1)


class ItemSelectionStrategy(ABC):
    """Abstract base class for CAT item selection strategies.

    Item selection strategies determine which item to administer next
    based on the current ability estimate and available item pool.
    """

    @abstractmethod
    def select_item(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> int:
        """Select the next item to administer.

        Parameters
        ----------
        model : BaseItemModel
            The fitted IRT model containing item parameters.
        theta : float
            Current ability estimate.
        available_items : set[int]
            Set of item indices that can still be administered.
        administered_items : list[int] | None
            List of already administered item indices.
        responses : list[int] | None
            List of responses to administered items.

        Returns
        -------
        int
            Index of the selected item.
        """
        pass

    def get_item_criteria(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> dict[int, float]:
        """Get selection criterion values for all available items.

        Parameters
        ----------
        model : BaseItemModel
            The fitted IRT model.
        theta : float
            Current ability estimate.
        available_items : set[int]
            Set of available item indices.
        administered_items : list[int] | None
            List of already administered item indices.
        responses : list[int] | None
            Responses to the administered items.

        Returns
        -------
        dict[int, float]
            Dictionary mapping item indices to criterion values.
        """
        criteria = {}
        theta_arr = np.array([[theta]])
        for item_idx in available_items:
            criteria[item_idx] = self._compute_criterion(model, theta_arr, item_idx)
        return criteria

    def _compute_criterion(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> float:
        """Compute the selection criterion for a single item.

        Parameters
        ----------
        model : BaseItemModel
            The fitted IRT model.
        theta : NDArray[np.float64]
            Ability estimate array of shape (1, n_factors).
        item_idx : int
            Index of the item.

        Returns
        -------
        float
            Criterion value (higher = more desirable).
        """
        return 0.0


@_register_native_defaults
class MaxFisherInformation(ItemSelectionStrategy):
    """Maximum Fisher Information (MFI) item selection.

    Selects the item that provides the maximum Fisher information
    at the current ability estimate. This is the most common
    item selection method in CAT. Ties go to the lowest item index.

    References
    ----------
    Lord, F. M. (1980). Applications of item response theory to
    practical testing problems. Lawrence Erlbaum Associates.
    """

    def select_item(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> int:
        if not available_items:
            raise ValueError("No available items to select from")

        criteria = self.get_item_criteria(model, theta, available_items)
        return max(criteria, key=criteria.__getitem__)

    def get_item_criteria(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> dict[int, float]:
        """Get Fisher information with one model call when supported.

        Items are returned in increasing index order, so the first maximum
        does not depend on set iteration order.
        """
        theta_arr = np.array([[theta]])
        items = sorted(available_items)

        # Polytomous models define information(theta) as total test
        # information rather than an item-wise array.
        if not model.is_polytomous:
            information = np.asarray(model.information(theta_arr))
            if information.size == model.n_items:
                item_information = information.reshape(model.n_items)
                return {
                    item_idx: float(item_information[item_idx]) for item_idx in items
                }

        return {
            item_idx: self._compute_criterion(model, theta_arr, item_idx)
            for item_idx in items
        }

    def _compute_criterion(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> float:
        info = model.information(theta, item_idx=item_idx)
        return float(info.sum())


class MaxExpectedInformation(ItemSelectionStrategy):
    """Maximum Expected Information (MEI) item selection.

    Selects the item that maximizes expected posterior information,
    accounting for uncertainty in the ability estimate by integrating
    over possible responses.

    Parameters
    ----------
    n_quadpts : int, optional
        Number of quadrature points for integration. Default is 21.
    theta_bounds : tuple[float, float], optional
        Bounds for posterior integration. Default is (-4.0, 4.0).

    References
    ----------
    van der Linden, W. J. (1998). Bayesian item selection criteria
    for adaptive testing. Psychometrika, 63(2), 201-216.
    """

    def __init__(
        self,
        n_quadpts: int = 21,
        theta_bounds: tuple[float, float] = (-4.0, 4.0),
    ) -> None:
        if (
            isinstance(n_quadpts, bool)
            or not isinstance(n_quadpts, (int, np.integer))
            or n_quadpts < 5
        ):
            raise ValueError("n_quadpts must be an integer of at least 5")

        try:
            lower, upper = theta_bounds
            lower = float(lower)
            upper = float(upper)
        except (TypeError, ValueError) as exc:
            raise ValueError("theta_bounds must contain two finite values") from exc
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(
                "theta_bounds must contain finite values with lower < upper"
            )

        self.n_quadpts = int(n_quadpts)
        self.theta_bounds = (lower, upper)

        raw_nodes, raw_weights = np.polynomial.legendre.leggauss(self.n_quadpts)
        half_width = (upper - lower) / 2.0
        midpoint = (upper + lower) / 2.0
        self._theta_nodes = midpoint + half_width * raw_nodes
        integration_weights = half_width * raw_weights
        self._log_prior_mass = (
            np.log(integration_weights)
            - 0.5 * np.square(self._theta_nodes)
            - 0.5 * np.log(2.0 * np.pi)
        )

    def select_item(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> int:
        if not available_items:
            raise ValueError("No available items to select from")

        criteria = self.get_item_criteria(
            model,
            theta,
            available_items,
            administered_items=administered_items,
            responses=responses,
        )
        return max(criteria, key=criteria.__getitem__)

    def get_item_criteria(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> dict[int, float]:
        """Compute history-aware expected information for available items.

        Dichotomous item banks use a fixed number of model calls per
        selection: response probabilities at the current ability and at the
        posterior nodes, plus item information at every hypothetical posterior
        mean in bounded row blocks. Polytomous banks evaluate each candidate's
        own curves, but each administered item's information is evaluated once
        for all hypothetical abilities.
        """
        if model.n_factors != 1:
            raise ValueError("MEI only supports unidimensional models")
        if not np.isfinite(theta):
            raise ValueError("theta must be finite")

        administered, observed_responses = self._validate_history(
            model, administered_items, responses
        )
        items = _validate_candidate_items(model, available_items)
        if not items:
            return {}
        history_log_mass = self._history_log_mass(
            model, administered, observed_responses
        )

        values = None
        if not model.is_polytomous:
            values = self._binary_expected_information(
                model, float(theta), items, administered, history_log_mass
            )
        if values is None:
            values = self._itemwise_expected_information(
                model, float(theta), items, administered, history_log_mass
            )
        return dict(zip(items, values.tolist(), strict=True))

    @staticmethod
    def _validate_history(
        model: BaseItemModel,
        administered_items: list[int] | None,
        responses: list[int] | None,
    ) -> tuple[list[int], list[int]]:
        if administered_items is None and responses is None:
            return [], []
        if administered_items is None and responses is not None and len(responses) == 0:
            return [], []
        if (
            responses is None
            and administered_items is not None
            and len(administered_items) == 0
        ):
            return [], []
        if administered_items is None or responses is None:
            raise ValueError(
                "administered_items and responses must be provided together"
            )

        administered = list(administered_items)
        observed_responses = list(responses)
        if len(administered) != len(observed_responses):
            raise ValueError("administered_items and responses must have equal length")

        normalized_items: list[int] = []
        normalized_responses: list[int] = []
        for item_idx, response in zip(administered, observed_responses, strict=True):
            if isinstance(item_idx, bool) or not isinstance(
                item_idx, (int, np.integer)
            ):
                raise ValueError("administered item indices must be integers")
            item_idx = int(item_idx)
            if item_idx < 0 or item_idx >= model.n_items:
                raise ValueError(f"administered item {item_idx} is out of range")
            if isinstance(response, bool) or not isinstance(
                response, (int, np.integer)
            ):
                raise ValueError("each response must be an integer category")
            response = int(response)
            n_categories = model._n_categories[item_idx] if model.is_polytomous else 2
            if response < 0 or response >= n_categories:
                raise ValueError(
                    f"response {response} is outside the category range for item {item_idx}"
                )
            normalized_items.append(item_idx)
            normalized_responses.append(response)

        if len(set(normalized_items)) != len(normalized_items):
            raise ValueError("administered_items must not contain duplicates")
        return normalized_items, normalized_responses

    def _history_log_mass(
        self,
        model: BaseItemModel,
        administered_items: list[int],
        responses: list[int],
    ) -> NDArray[np.float64]:
        if not administered_items:
            return self._log_prior_mass.copy()

        response_matrix = np.full((1, model.n_items), -1, dtype=np.int_)
        response_matrix[0, administered_items] = responses
        log_likelihood = np.asarray(
            model.log_likelihood_batch(
                response_matrix,
                self._theta_nodes[:, None],
            ),
            dtype=np.float64,
        ).ravel()
        return log_likelihood + self._log_prior_mass

    def _posterior_means(
        self,
        history_log_mass: NDArray[np.float64],
        node_probabilities: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return posterior means after each response, reducing over axis 0.

        ``node_probabilities`` holds response probabilities with the
        quadrature nodes on the first axis; the result drops that axis.
        """
        clipped = np.clip(node_probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON)
        extra_axes = (slice(None),) + (None,) * (clipped.ndim - 1)
        log_mass = history_log_mass[extra_axes] + np.log(clipped)
        log_mass -= np.max(log_mass, axis=0, keepdims=True)
        posterior_mass = np.exp(log_mass)
        posterior_mass /= posterior_mass.sum(axis=0, keepdims=True)
        return np.moveaxis(posterior_mass, 0, -1) @ self._theta_nodes

    def _binary_expected_information(
        self,
        model: BaseItemModel,
        theta: float,
        items: list[int],
        administered: list[int],
        history_log_mass: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """Evaluate every dichotomous candidate with bulk model calls.

        Return None when the model's bulk output is not one value per item,
        so customized models keep the item-wise evaluation.
        """
        n_items = model.n_items
        candidates = np.asarray(items, dtype=np.intp)
        current = np.asarray(model.probability(np.array([[theta]])), dtype=np.float64)
        nodes = np.asarray(
            model.probability(self._theta_nodes[:, None]), dtype=np.float64
        )
        if current.size != n_items or nodes.size != self.n_quadpts * n_items:
            return None

        p_current = current.reshape(n_items)[candidates]
        response_probabilities = np.stack((1.0 - p_current, p_current), axis=1)
        p_nodes = nodes.reshape(self.n_quadpts, n_items)[:, candidates]
        node_probabilities = np.stack((1.0 - p_nodes, p_nodes), axis=2)
        # Two hypothetical abilities per candidate: rows 2j and 2j + 1.
        hypothetical_theta = self._posterior_means(
            history_log_mass, node_probabilities
        ).reshape(-1)

        own_columns = np.repeat(candidates, 2)
        test_information = np.empty(hypothetical_theta.size, dtype=np.float64)
        block_size = max(1, _MAX_INFORMATION_VALUES // n_items)
        for start in range(0, hypothetical_theta.size, block_size):
            stop = min(start + block_size, hypothetical_theta.size)
            information = np.asarray(
                model.information(hypothetical_theta[start:stop, None]),
                dtype=np.float64,
            )
            if information.size != (stop - start) * n_items:
                return None
            information = information.reshape(stop - start, n_items)
            block = information[:, administered].sum(axis=1)
            block += information[np.arange(stop - start), own_columns[start:stop]]
            test_information[start:stop] = block

        return np.einsum(
            "jr,jr->j", response_probabilities, test_information.reshape(-1, 2)
        )

    def _itemwise_expected_information(
        self,
        model: BaseItemModel,
        theta: float,
        items: list[int],
        administered: list[int],
        history_log_mass: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate candidates through per-item model calls.

        Each candidate's response curves are evaluated separately, while each
        administered item's information is evaluated once at the hypothetical
        abilities of all candidates.
        """
        response_probabilities = []
        hypothetical_theta = []
        for item_idx in items:
            response_probabilities.append(
                self._response_probabilities(model, theta, item_idx)
            )
            hypothetical_theta.append(
                self._posterior_means(
                    history_log_mass,
                    self._node_response_probabilities(model, item_idx),
                )
            )
        counts = [len(values) for values in hypothetical_theta]
        all_theta = np.concatenate(hypothetical_theta)[:, None]

        test_information = np.zeros(len(all_theta), dtype=np.float64)
        for item_idx in administered:
            test_information += _information_at(model, all_theta, item_idx)

        offsets = np.cumsum([0, *counts])
        criteria = np.empty(len(items), dtype=np.float64)
        for position, item_idx in enumerate(items):
            rows = slice(offsets[position], offsets[position + 1])
            information = test_information[rows] + _information_at(
                model, all_theta[rows], item_idx
            )
            criteria[position] = response_probabilities[position] @ information
        return criteria

    @staticmethod
    def _response_probabilities(
        model: BaseItemModel,
        theta: float,
        item_idx: int,
    ) -> NDArray[np.float64]:
        probabilities = np.asarray(
            model.probability(np.array([[theta]]), item_idx=item_idx),
            dtype=np.float64,
        ).ravel()
        if model.is_polytomous:
            return probabilities
        probability_correct = probabilities[0]
        return np.array([1.0 - probability_correct, probability_correct])

    def _node_response_probabilities(
        self,
        model: BaseItemModel,
        item_idx: int,
    ) -> NDArray[np.float64]:
        probabilities = np.asarray(
            model.probability(self._theta_nodes[:, None], item_idx=item_idx),
            dtype=np.float64,
        )
        if model.is_polytomous:
            return probabilities.reshape(self.n_quadpts, -1)
        probability_correct = probabilities.ravel()
        return np.column_stack((1.0 - probability_correct, probability_correct))

    def _compute_criterion(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> float:
        # Bypass overrides: a subclass's get_item_criteria may delegate here.
        criteria = MaxExpectedInformation.get_item_criteria(
            self, model, float(theta[0, 0]), {item_idx}
        )
        return criteria[item_idx]


class KullbackLeibler(ItemSelectionStrategy):
    """Kullback-Leibler (KL) divergence item selection.

    Selects the item that maximizes the expected KL divergence
    between the response distributions at the current theta
    and neighboring theta values.

    Parameters
    ----------
    delta : float, optional
        Half-width of the interval for KL integration. Default is 0.1.
    n_points : int, optional
        Number of points for numerical integration. Default is 5.

    References
    ----------
    Chang, H.-H., & Ying, Z. (1996). A global information approach
    to computerized adaptive testing. Applied Psychological
    Measurement, 20(3), 213-229.
    """

    def __init__(self, delta: float = 0.1, n_points: int = 5):
        if isinstance(delta, (bool, np.bool_)) or not isinstance(delta, Real):
            raise ValueError("delta must be a finite positive number")
        delta_value = float(delta)
        if not np.isfinite(delta_value) or delta_value <= 0.0:
            raise ValueError("delta must be a finite positive number")
        if (
            isinstance(n_points, (bool, np.bool_))
            or not isinstance(n_points, Integral)
            or n_points < 2
        ):
            raise ValueError("n_points must be an integer of at least 2")

        self.delta = delta_value
        self.n_points = int(n_points)
        offsets = np.linspace(-self.delta, self.delta, self.n_points)
        self._neighbor_offsets = offsets[offsets != 0.0]

    def select_item(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> int:
        if not available_items:
            raise ValueError("No available items to select from")

        criteria = self.get_item_criteria(model, theta, available_items)
        return max(criteria, key=lambda item_idx: (criteria[item_idx], -item_idx))

    def get_item_criteria(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> dict[int, float]:
        """Evaluate the KL grid once for a dichotomous item bank."""
        if model.n_factors != 1:
            raise ValueError("KL selection only supports unidimensional models")

        item_indices = _validate_candidate_items(model, available_items)
        if not item_indices:
            return {}

        theta_values = self._evaluation_thetas(theta)
        if not model.is_polytomous:
            probabilities = np.asarray(
                model.probability(theta_values),
                dtype=np.float64,
            ).reshape(len(theta_values), model.n_items)
            candidate_probabilities = probabilities[:, item_indices]
            criteria = self._mean_bernoulli_kl(candidate_probabilities)
            return {
                item_idx: float(criterion)
                for item_idx, criterion in zip(
                    item_indices,
                    criteria,
                    strict=True,
                )
            }

        return {
            item_idx: self._mean_categorical_kl(
                np.asarray(
                    model.probability(theta_values, item_idx=item_idx),
                    dtype=np.float64,
                )
            )
            for item_idx in item_indices
        }

    def _evaluation_thetas(self, theta: float) -> NDArray[np.float64]:
        """Return the current theta followed by every nonzero grid neighbor."""
        if isinstance(theta, (bool, np.bool_)) or not isinstance(theta, Real):
            raise ValueError("theta must be finite")
        theta_value = float(theta)
        if not np.isfinite(theta_value):
            raise ValueError("theta must be finite")

        with np.errstate(over="ignore", invalid="ignore"):
            neighbors = theta_value + self._neighbor_offsets
        if not np.all(np.isfinite(neighbors)):
            raise ValueError("theta neighborhood must be finite")
        return np.concatenate(([theta_value], neighbors))[:, None]

    @staticmethod
    def _validate_probabilities(
        probabilities: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Reject invalid model output before evaluating divergence."""
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise ValueError("model probabilities must be finite values in [0, 1]")
        return probabilities

    def _mean_bernoulli_kl(
        self,
        probabilities: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return mean Bernoulli KL divergence for each item column."""
        probabilities = self._validate_probabilities(probabilities)
        current = np.clip(
            probabilities[0],
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        neighbors = np.clip(
            probabilities[1:],
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        complement = 1.0 - current
        divergences = current * np.log(current / neighbors) + complement * np.log(
            complement / (1.0 - neighbors)
        )
        return np.maximum(np.mean(divergences, axis=0), 0.0)

    def _mean_categorical_kl(
        self,
        probabilities: NDArray[np.float64],
    ) -> float:
        """Return mean categorical KL divergence across neighboring thetas."""
        probabilities = self._validate_probabilities(probabilities)
        probabilities = probabilities.reshape(probabilities.shape[0], -1)
        probabilities = np.clip(probabilities, PROB_EPSILON, 1.0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        current = probabilities[0]
        divergences = np.sum(
            current * np.log(current / probabilities[1:]),
            axis=1,
        )
        return max(float(np.mean(divergences)), 0.0)

    def _compute_kl_info(
        self,
        model: BaseItemModel,
        theta: float,
        item_idx: int,
    ) -> float:
        """Compute KL information for an item at theta."""
        theta_values = self._evaluation_thetas(theta)
        probabilities = np.asarray(
            model.probability(theta_values, item_idx=item_idx),
            dtype=np.float64,
        )
        if model.is_polytomous:
            return self._mean_categorical_kl(probabilities)
        return float(self._mean_bernoulli_kl(probabilities.reshape(-1, 1))[0])

    def _kl_divergence(
        self,
        p: NDArray[np.float64],
        q: NDArray[np.float64],
    ) -> float:
        """Compute KL divergence D(p || q)."""
        p = self._validate_probabilities(np.asarray(p, dtype=np.float64).ravel())
        q = self._validate_probabilities(np.asarray(q, dtype=np.float64).ravel())
        if p.shape != q.shape:
            raise ValueError("probability vectors must have the same shape")

        if len(p) == 1:
            probabilities = np.array([[p[0]], [q[0]]], dtype=np.float64)
            return float(self._mean_bernoulli_kl(probabilities)[0])
        else:
            probabilities = np.vstack((p, q))
            return self._mean_categorical_kl(probabilities)

    def _compute_criterion(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> float:
        return self._compute_kl_info(model, float(theta[0, 0]), item_idx)


def _item_location(model: BaseItemModel, item_idx: int) -> float:
    """Return an item's difficulty, or its mean threshold, defaulting to 0."""
    params = model.get_item_parameters(item_idx)
    if "difficulty" in params:
        return float(np.mean(params["difficulty"]))
    if "thresholds" in params:
        return float(np.mean(params["thresholds"]))
    return 0.0


class UrryRule(ItemSelectionStrategy):
    """Urry's rule for item selection.

    Selects the item with difficulty parameter closest to the
    current ability estimate. Simple and computationally efficient.

    References
    ----------
    Urry, V. W. (1977). Tailored testing: A successful application
    of latent trait theory. Journal of Educational Measurement,
    14(2), 181-196.
    """

    def select_item(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> int:
        if not available_items:
            raise ValueError("No available items to select from")

        best_item = -1
        min_diff = np.inf

        for item_idx in available_items:
            diff = abs(theta - _item_location(model, item_idx))
            if diff < min_diff:
                min_diff = diff
                best_item = item_idx

        return best_item

    def _compute_criterion(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> float:
        return -abs(float(theta[0, 0]) - _item_location(model, item_idx))


class RandomSelection(ItemSelectionStrategy):
    """Random item selection.

    Randomly selects an item from the available pool.
    Useful as a baseline or for initial items in CAT.

    Parameters
    ----------
    seed : int | None, optional
        Random seed for reproducibility. Default is None.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def select_item(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> int:
        if not available_items:
            raise ValueError("No available items to select from")

        items_list = list(available_items)
        return items_list[self.rng.integers(len(items_list))]

    def get_item_criteria(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
    ) -> dict[int, float]:
        """Return independent uniform scores from this strategy's generator.

        Ranking these scores, as randomesque exposure control does, yields a
        seeded uniformly random choice instead of a fixed item order.
        """
        items = sorted(available_items)
        return dict(zip(items, self.rng.random(len(items)).tolist(), strict=True))


class AStratified(ItemSelectionStrategy):
    """A-stratified item selection.

    Divides the pool into strata of increasing mean discrimination and moves
    through them as the test progresses. A test of ``test_length`` items
    spends ``test_length / n_strata`` items in each stratum, starting with the
    least discriminating items, as in Chang and Ying (1999). Saving highly
    discriminating items for later stages, when the ability estimate is more
    accurate, spreads exposure across the pool.

    Within the current stratum, items are ranked by Fisher information at the
    current ability (``within="MFI"``) or by the closeness of their
    difficulty to the current ability (``within="b-matching"``, the rule of
    the original paper). When the current stratum has no available items, the
    next non-empty later stratum is used, then the whole available pool.

    Parameters
    ----------
    n_strata : int, optional
        Number of discrimination strata. Default is 3.
    test_length : int | None, optional
        Planned test length that schedules the strata, capped at the pool
        size. When None, CATEngine supplies its ``max_items`` (or the pool
        size without one) on every selection, and standalone calls use the
        pool size. Variable-length tests should set ``max_items`` or this
        value; otherwise later strata are reached only near pool exhaustion.
    within : {"MFI", "b-matching"}, optional
        Ranking rule within a stratum. Default is "MFI".

    References
    ----------
    Chang, H.-H., & Ying, Z. (1999). a-Stratified multistage
    computerized adaptive testing. Applied Psychological
    Measurement, 23(3), 211-222.
    """

    def __init__(
        self,
        n_strata: int = 3,
        test_length: int | None = None,
        within: Literal["MFI", "b-matching"] = "MFI",
    ) -> None:
        if (
            isinstance(n_strata, (bool, np.bool_))
            or not isinstance(n_strata, Integral)
            or n_strata < 1
        ):
            raise ValueError("n_strata must be a positive integer")
        if test_length is not None and (
            isinstance(test_length, (bool, np.bool_))
            or not isinstance(test_length, Integral)
            or test_length < 1
        ):
            raise ValueError("test_length must be a positive integer or None")
        rules = {"mfi": "MFI", "b-matching": "b-matching"}
        rule = rules.get(str(within).strip().lower().replace("_", "-"))
        if rule is None:
            raise ValueError("within must be 'MFI' or 'b-matching'")

        self.n_strata: int = int(n_strata)
        self.test_length: int | None = None if test_length is None else int(test_length)
        self.within: str = rule
        self._strata: list[set[int]] | None = None
        # The pool the strata were built for; identity, not id(), so a new
        # model allocated at a collected model's address is restratified.
        self._strata_model: BaseItemModel | None = None
        self._strata_size: int = 0

    def _initialize_strata(self, model: BaseItemModel) -> list[set[int]]:
        """Partition items into strata of increasing discrimination."""
        discriminations = []
        for i in range(model.n_items):
            params = model.get_item_parameters(i)
            if "discrimination" in params:
                a = params["discrimination"]
                if isinstance(a, np.ndarray):
                    a = float(np.mean(a))
                discriminations.append((i, a))
            else:
                discriminations.append((i, 1.0))

        discriminations.sort(key=lambda x: x[1])

        n_items = len(discriminations)
        items_per_stratum = n_items // self.n_strata
        remainder = n_items % self.n_strata

        strata = []
        start = 0
        for s in range(self.n_strata):
            end = start + items_per_stratum + (1 if s < remainder else 0)
            strata.append({discriminations[i][0] for i in range(start, end)})
            start = end
        self._strata = strata
        self._strata_model = model
        self._strata_size = model.n_items
        return strata

    def current_stratum(
        self,
        model: BaseItemModel,
        n_administered: int,
        *,
        test_length: int | None = None,
    ) -> int:
        """Return the scheduled stratum index after ``n_administered`` items.

        Parameters
        ----------
        model : BaseItemModel
            The fitted IRT model.
        n_administered : int
            Number of items already administered.
        test_length : int | None, optional
            Planned test length used when the strategy has none configured.
            Defaults to the pool size, which also caps any planned length.

        Returns
        -------
        int
            Zero-based stratum index, ordered by increasing discrimination.
        """
        length = min(self.test_length or test_length or model.n_items, model.n_items)
        return min(self.n_strata - 1, n_administered * self.n_strata // length)

    def _stratum_candidates(
        self,
        model: BaseItemModel,
        available_items: set[int],
        administered_items: list[int] | None,
        test_length: int | None,
    ) -> set[int]:
        """Return the available items of the scheduled or next non-empty stratum."""
        strata = self._strata
        if (
            strata is None
            or self._strata_model is not model
            or self._strata_size != model.n_items
        ):
            strata = self._initialize_strata(model)

        n_administered = len(administered_items) if administered_items else 0
        stage = self.current_stratum(model, n_administered, test_length=test_length)
        for stratum in strata[stage:]:
            candidates = available_items & stratum
            if candidates:
                return candidates
        return set(available_items)

    def get_item_criteria(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
        *,
        test_length: int | None = None,
    ) -> dict[int, float]:
        """Rank only the scheduled stratum's available items.

        Items outside that stratum are omitted, so ranking-based exposure
        control such as randomesque selection stays within it.
        """
        candidates = self._stratum_candidates(
            model, set(available_items), administered_items, test_length
        )
        if self.within == "MFI":
            return MaxFisherInformation().get_item_criteria(model, theta, candidates)
        return {
            item_idx: -abs(theta - _item_location(model, item_idx))
            for item_idx in sorted(candidates)
        }

    def select_item(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
        *,
        test_length: int | None = None,
    ) -> int:
        if not available_items:
            raise ValueError("No available items to select from")

        # Subclasses may still override get_item_criteria without test_length.
        options = {} if test_length is None else {"test_length": test_length}
        criteria = self.get_item_criteria(
            model, theta, available_items, administered_items, responses, **options
        )
        return max(sorted(criteria), key=criteria.__getitem__)


_SELECTION_STRATEGIES: dict[str, type[ItemSelectionStrategy]] = {
    "MFI": MaxFisherInformation,
    "MEI": MaxExpectedInformation,
    "KL": KullbackLeibler,
    "Urry": UrryRule,
    "random": RandomSelection,
    "a-stratified": AStratified,
}


def _selection_strategy_class(method: str) -> type[ItemSelectionStrategy]:
    """Resolve a strategy name, ignoring case, whitespace, and ``_`` vs ``-``."""
    normalized = method.strip().lower().replace("_", "-")
    for name, strategy_class in _SELECTION_STRATEGIES.items():
        if name.lower() == normalized:
            return strategy_class
    valid = ", ".join(_SELECTION_STRATEGIES)
    raise ValueError(f"Unknown selection method '{method}'. Valid options: {valid}")


def create_selection_strategy(
    method: str,
    **kwargs: Any,
) -> ItemSelectionStrategy:
    """Factory function to create item selection strategies.

    Parameters
    ----------
    method : str
        Selection method name. One of: "MFI", "MEI", "KL", "Urry",
        "random", "a-stratified". Names match regardless of case and
        surrounding whitespace, and ``"_"`` is accepted for ``"-"``, so
        ``"a_stratified"`` and ``"urry"`` also work.
    **kwargs
        Additional keyword arguments passed to the strategy constructor.

    Returns
    -------
    ItemSelectionStrategy
        The requested item selection strategy.

    Raises
    ------
    ValueError
        If the method is not recognized.
    """
    return _selection_strategy_class(method)(**kwargs)
