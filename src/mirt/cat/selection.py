"""Item selection strategies for computerized adaptive testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


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


class MaxFisherInformation(ItemSelectionStrategy):
    """Maximum Fisher Information (MFI) item selection.

    Selects the item that provides the maximum Fisher information
    at the current ability estimate. This is the most common
    item selection method in CAT.

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
        """Get Fisher information with one model call when supported."""
        theta_arr = np.array([[theta]])

        # Polytomous models define information(theta) as total test
        # information rather than an item-wise array.
        if not model.is_polytomous:
            information = np.asarray(model.information(theta_arr))
            if information.size == model.n_items:
                item_information = information.reshape(model.n_items)
                return {
                    item_idx: float(item_information[item_idx])
                    for item_idx in available_items
                }

        return {
            item_idx: self._compute_criterion(model, theta_arr, item_idx)
            for item_idx in available_items
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
        """Compute history-aware expected information for available items."""
        if model.n_factors != 1:
            raise ValueError("MEI only supports unidimensional models")
        if not np.isfinite(theta):
            raise ValueError("theta must be finite")

        administered, observed_responses = self._validate_history(
            model, administered_items, responses
        )
        history_log_mass = self._history_log_mass(
            model, administered, observed_responses
        )
        return {
            item_idx: self._compute_expected_information(
                model,
                theta,
                item_idx,
                administered,
                history_log_mass,
            )
            for item_idx in available_items
        }

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

    def _compute_expected_information(
        self,
        model: BaseItemModel,
        theta: float,
        item_idx: int,
        administered_items: list[int],
        history_log_mass: NDArray[np.float64],
    ) -> float:
        """Compute information after each hypothetical response to an item."""
        current_probabilities = self._response_probabilities(model, theta, item_idx)
        node_probabilities = self._node_response_probabilities(model, item_idx)
        clipped_probabilities = np.clip(
            node_probabilities,
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )

        log_mass = history_log_mass[:, None] + np.log(clipped_probabilities)
        log_mass -= np.max(log_mass, axis=0, keepdims=True)
        posterior_mass = np.exp(log_mass)
        posterior_mass /= posterior_mass.sum(axis=0, keepdims=True)
        hypothetical_theta = posterior_mass.T @ self._theta_nodes

        theta_values = hypothetical_theta[:, None]
        test_information = np.zeros(len(current_probabilities), dtype=np.float64)
        for provisional_item in (*administered_items, item_idx):
            information = np.asarray(
                model.information(theta_values, item_idx=provisional_item),
                dtype=np.float64,
            )
            test_information += information.reshape(len(current_probabilities), -1).sum(
                axis=1
            )

        return float(current_probabilities @ test_information)

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
        history_log_mass = self._history_log_mass(model, [], [])
        return self._compute_expected_information(
            model,
            float(theta[0, 0]),
            item_idx,
            [],
            history_log_mass,
        )


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
        self.delta = delta
        self.n_points = n_points

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
        best_kl = -np.inf

        for item_idx in available_items:
            kl = self._compute_kl_info(model, theta, item_idx)

            if kl > best_kl:
                best_kl = kl
                best_item = item_idx

        return best_item

    def _compute_kl_info(
        self,
        model: BaseItemModel,
        theta: float,
        item_idx: int,
    ) -> float:
        """Compute KL information for an item at theta."""
        theta_arr = np.array([[theta]])
        prob_theta = model.probability(theta_arr, item_idx=item_idx)

        theta_points = np.linspace(
            theta - self.delta, theta + self.delta, self.n_points
        )

        kl_sum = 0.0
        for t in theta_points:
            if t == theta:
                continue
            t_arr = np.array([[t]])
            prob_t = model.probability(t_arr, item_idx=item_idx)

            kl = self._kl_divergence(prob_theta, prob_t)
            kl_sum += kl

        return kl_sum / (self.n_points - 1)

    def _kl_divergence(
        self,
        p: NDArray[np.float64],
        q: NDArray[np.float64],
    ) -> float:
        """Compute KL divergence D(p || q)."""
        p = np.clip(p.ravel(), PROB_EPSILON, 1 - PROB_EPSILON)
        q = np.clip(q.ravel(), PROB_EPSILON, 1 - PROB_EPSILON)

        if len(p) == 1:
            p_full = np.array([p[0], 1 - p[0]])
            q_full = np.array([q[0], 1 - q[0]])
        else:
            p_full = p
            q_full = q

        return float(np.sum(p_full * np.log(p_full / q_full)))

    def _compute_criterion(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> float:
        return self._compute_kl_info(model, float(theta[0, 0]), item_idx)


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
            params = model.get_item_parameters(item_idx)

            if "difficulty" in params:
                b = params["difficulty"]
            elif "thresholds" in params:
                b = np.mean(params["thresholds"])
            else:
                b = 0.0

            diff = abs(theta - b)
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
        params = model.get_item_parameters(item_idx)
        if "difficulty" in params:
            b = params["difficulty"]
        elif "thresholds" in params:
            b = np.mean(params["thresholds"])
        else:
            b = 0.0
        return -abs(float(theta[0, 0]) - b)


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

    def _compute_criterion(
        self,
        model: BaseItemModel,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> float:
        return 0.0


class AStratified(ItemSelectionStrategy):
    """A-stratified item selection with content balancing.

    Divides items into strata based on discrimination parameters
    and selects from appropriate strata as the test progresses.
    Early items come from low-discrimination strata, later items
    from high-discrimination strata.

    Parameters
    ----------
    n_strata : int, optional
        Number of discrimination strata. Default is 3.

    References
    ----------
    Chang, H.-H., & Ying, Z. (1999). a-Stratified multistage
    computerized adaptive testing. Applied Psychological
    Measurement, 23(3), 211-222.
    """

    def __init__(self, n_strata: int = 3):
        self.n_strata = n_strata
        self._strata: list[set[int]] | None = None

    def _initialize_strata(self, model: BaseItemModel) -> None:
        """Initialize item strata based on discrimination."""
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

        self._strata = []
        start = 0
        for s in range(self.n_strata):
            end = start + items_per_stratum + (1 if s < remainder else 0)
            stratum_items = {discriminations[i][0] for i in range(start, end)}
            self._strata.append(stratum_items)
            start = end

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

        if self._strata is None:
            self._initialize_strata(model)

        n_administered = len(administered_items) if administered_items else 0

        stratum_idx = min(
            n_administered // (model.n_items // self.n_strata // 2 + 1),
            self.n_strata - 1,
        )

        for s in range(stratum_idx, self.n_strata):
            stratum_available = available_items & self._strata[s]
            if stratum_available:
                mfi = MaxFisherInformation()
                return mfi.select_item(model, theta, stratum_available)

        mfi = MaxFisherInformation()
        return mfi.select_item(model, theta, available_items)


def create_selection_strategy(
    method: str,
    **kwargs: Any,
) -> ItemSelectionStrategy:
    """Factory function to create item selection strategies.

    Parameters
    ----------
    method : str
        Selection method name. One of: "MFI", "MEI", "KL", "Urry",
        "random", "a-stratified".
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
    strategies = {
        "MFI": MaxFisherInformation,
        "MEI": MaxExpectedInformation,
        "KL": KullbackLeibler,
        "Urry": UrryRule,
        "random": RandomSelection,
        "a-stratified": AStratified,
    }

    method_upper = (
        method.upper() if method not in ("random", "a-stratified") else method
    )

    if method_upper not in strategies and method not in strategies:
        valid = ", ".join(strategies.keys())
        raise ValueError(f"Unknown selection method '{method}'. Valid options: {valid}")

    if method_upper in strategies:
        strategy_class = strategies[method_upper]
    else:
        strategy_class = strategies[method]
    return strategy_class(**kwargs)
