"""CAT engine for orchestrating computerized adaptive testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from mirt._rust_backend import (
    RUST_AVAILABLE,
)
from mirt._rust_backend import (
    cat_conditional_mse as rust_cat_conditional_mse,
)
from mirt._rust_backend import (
    cat_simulate_batch as rust_cat_simulate_batch,
)
from mirt.cat.content import (
    ContentConstraint,
    NoContentConstraint,
)
from mirt.cat.exposure import (
    ExposureControl,
    NoExposureControl,
    Randomesque,
    create_exposure_control,
)
from mirt.cat.results import CATResult, CATState
from mirt.cat.selection import (
    ItemSelectionStrategy,
    MaxFisherInformation,
    create_selection_strategy,
)
from mirt.cat.stopping import (
    CombinedStop,
    MaxItemsStop,
    StandardErrorStop,
    StoppingRule,
    create_stopping_rule,
)

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class CATEngine:
    """Engine for computerized adaptive testing.

    Orchestrates the adaptive testing process including item selection,
    ability estimation, stopping rule evaluation, and constraint handling.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model containing item parameters.
    item_selection : ItemSelectionStrategy | str, optional
        Item selection strategy or name. Default is "MFI".
        Options: "MFI", "MEI", "KL", "Urry", "random", "a-stratified".
    stopping_rule : StoppingRule | str, optional
        Stopping rule or name. Default is "SE".
        Options: "SE", "max_items", "theta_change", "classification".
    scoring_method : {"EAP", "MAP", "ML", "WLE"}, optional
        Method for ability estimation. Default is "EAP".
    initial_theta : float, optional
        Initial ability estimate. Default is 0.0.
    se_threshold : float, optional
        Standard error threshold for SE stopping rule. Default is 0.3.
    max_items : int | None, optional
        Maximum number of items to administer. Default is None.
    min_items : int, optional
        Minimum number of items before stopping. Default is 1.
    exposure_control : ExposureControl | str | None, optional
        Exposure control method. Default is None (no control).
    content_constraint : ContentConstraint | None, optional
        Content balancing constraint. Default is None (no constraint).
    n_quadpts : int, optional
        Number of quadrature points for EAP scoring. Default is 21.
    theta_bounds : tuple[float, float], optional
        Bounds for ability estimation. Default is (-4.0, 4.0).
    seed : int | None, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from mirt import fit_mirt
    >>> from mirt.cat import CATEngine
    >>> result = fit_mirt(data, model="2PL")
    >>> cat = CATEngine(result.model, se_threshold=0.3, max_items=20)
    >>> state = cat.get_current_state()
    >>> while not state.is_complete:
    ...     response = get_response(state.next_item)  # User provides response
    ...     state = cat.administer_item(response)
    >>> print(f"Final theta: {state.theta:.2f}")
    """

    def __init__(
        self,
        model: BaseItemModel,
        item_selection: ItemSelectionStrategy | str = "MFI",
        stopping_rule: StoppingRule | str = "SE",
        scoring_method: Literal["EAP", "MAP", "ML", "WLE"] = "EAP",
        initial_theta: float = 0.0,
        se_threshold: float = 0.3,
        max_items: int | None = None,
        min_items: int = 1,
        exposure_control: ExposureControl | str | None = None,
        content_constraint: ContentConstraint | None = None,
        n_quadpts: int = 21,
        theta_bounds: tuple[float, float] = (-4.0, 4.0),
        seed: int | None = None,
    ):
        if not model.is_fitted:
            raise ValueError("Model must be fitted before use in CAT")

        self.model = model
        self.scoring_method = scoring_method
        self.initial_theta = initial_theta
        self.n_quadpts = n_quadpts
        self.theta_bounds = theta_bounds
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Initialize item selection strategy
        if isinstance(item_selection, str):
            self._selection = create_selection_strategy(item_selection)
        else:
            self._selection = item_selection

        # Initialize stopping rule
        if isinstance(stopping_rule, str):
            if stopping_rule == "SE":
                base_rule = StandardErrorStop(se_threshold)
            else:
                base_rule = create_stopping_rule(stopping_rule)
        else:
            base_rule = stopping_rule

        # Combine with max_items and min_items if specified
        rules = [base_rule]
        if max_items is not None:
            rules.append(MaxItemsStop(max_items))

        if len(rules) > 1 or min_items > 1:
            self._stopping = CombinedStop(rules, operator="or", min_items=min_items)
        else:
            self._stopping = base_rule

        # Initialize exposure control
        if exposure_control is None:
            self._exposure = NoExposureControl()
        elif isinstance(exposure_control, str):
            self._exposure = create_exposure_control(exposure_control, seed=seed)
        else:
            self._exposure = exposure_control

        # Initialize content constraint
        if content_constraint is None:
            self._content = NoContentConstraint()
        else:
            self._content = content_constraint

        # Initialize state
        self._current_theta = initial_theta
        self._current_se = float("inf")
        self._items_administered: list[int] = []
        self._responses: list[int] = []
        self._available_items: set[int] = set(range(model.n_items))
        self._theta_history: list[float] = []
        self._se_history: list[float] = []
        self._info_history: list[float] = []
        self._is_complete = False
        self._stopping_reason = ""

    def reset(self) -> None:
        """Reset the engine for a new examinee."""
        self._current_theta = self.initial_theta
        self._current_se = float("inf")
        self._items_administered = []
        self._responses = []
        self._available_items = set(range(self.model.n_items))
        self._theta_history = []
        self._se_history = []
        self._info_history = []
        self._is_complete = False
        self._stopping_reason = ""

        self._exposure.reset()
        self._content.reset()

        if hasattr(self._stopping, "reset"):
            self._stopping.reset()

    def get_current_state(self) -> CATState:
        """Get the current state of the CAT session.

        Returns
        -------
        CATState
            Current state including theta, SE, items administered.
        """
        next_item = None
        if not self._is_complete and self._available_items:
            next_item = self._select_next_item()

        return CATState(
            theta=self._current_theta,
            standard_error=self._current_se,
            items_administered=list(self._items_administered),
            responses=list(self._responses),
            n_items=len(self._items_administered),
            is_complete=self._is_complete,
            next_item=next_item,
        )

    def select_next_item(self) -> int:
        """Select the next item to administer.

        Returns
        -------
        int
            Index of the selected item.

        Raises
        ------
        RuntimeError
            If the CAT session is complete or no items available.
        """
        if self._is_complete:
            raise RuntimeError("CAT session is complete")
        if not self._available_items:
            raise RuntimeError("No items available for selection")

        return self._select_next_item()

    def _select_next_item(self) -> int:
        """Internal item selection with constraint handling."""
        # Apply content constraints
        content_eligible = self._content.filter_items(
            self._available_items, self._items_administered
        )

        # Apply exposure control
        exposure_eligible = self._exposure.filter_items(
            content_eligible, self.model, self._current_theta
        )

        # If using randomesque, handle specially
        if isinstance(self._exposure, Randomesque):
            # Get ranked items
            criteria = self._selection.get_item_criteria(
                self.model, self._current_theta, exposure_eligible
            )
            ranked = sorted(criteria.items(), key=lambda x: x[1], reverse=True)
            return self._exposure.select_from_ranked(ranked)

        # Standard selection
        return self._selection.select_item(
            self.model,
            self._current_theta,
            exposure_eligible,
            self._items_administered,
            self._responses,
        )

    def administer_item(self, response: int) -> CATState:
        """Record a response and update the CAT state.

        Parameters
        ----------
        response : int
            Response to the administered item (0/1 for dichotomous,
            0..k for polytomous).

        Returns
        -------
        CATState
            Updated CAT state after processing the response.

        Raises
        ------
        RuntimeError
            If the CAT session is already complete.
        ValueError
            If no item has been selected to administer.
        """
        if self._is_complete:
            raise RuntimeError("CAT session is already complete")

        # Select item if not already selected
        if not hasattr(self, "_pending_item"):
            self._pending_item = self._select_next_item()

        item_idx = self._pending_item
        delattr(self, "_pending_item")

        # Record item information before update
        theta_arr = np.array([[self._current_theta]])
        item_info = float(self.model.information(theta_arr, item_idx=item_idx).sum())
        self._info_history.append(item_info)

        # Update administered items
        self._items_administered.append(item_idx)
        self._responses.append(response)
        self._available_items.discard(item_idx)

        # Update exposure control
        self._exposure.update(item_idx)

        # Update ability estimate
        self._update_theta()

        # Record history
        self._theta_history.append(self._current_theta)
        self._se_history.append(self._current_se)

        # Check stopping rule
        state = self.get_current_state()
        if self._stopping.should_stop(state):
            self._is_complete = True
            self._stopping_reason = self._stopping.get_reason()

        # Check if items exhausted
        if not self._available_items and not self._is_complete:
            self._is_complete = True
            self._stopping_reason = "Item pool exhausted"

        return self.get_current_state()

    def _update_theta(self) -> None:
        """Update ability estimate based on administered items."""
        from mirt.scoring import fscores

        # Build response array (all items, with -1 for non-administered)
        responses = np.full((1, self.model.n_items), -1, dtype=np.int_)
        for item_idx, resp in zip(self._items_administered, self._responses):
            responses[0, item_idx] = resp

        # Score with selected method
        try:
            result = fscores(
                self.model,
                responses,
                method=self.scoring_method,
                n_quadpts=self.n_quadpts,
                bounds=self.theta_bounds,
            )
            self._current_theta = float(result.theta.ravel()[0])
            self._current_se = float(result.standard_error.ravel()[0])
        except Exception:
            # Fallback to simple estimate on failure
            self._current_theta = self._simple_theta_estimate()
            self._current_se = self._simple_se_estimate()

    def _simple_theta_estimate(self) -> float:
        """Simple theta estimate based on proportion correct."""
        if not self._responses:
            return self.initial_theta

        # For dichotomous items
        p = np.mean(self._responses)
        p = np.clip(p, 0.01, 0.99)

        # Logit transform
        return float(np.log(p / (1 - p)))

    def _simple_se_estimate(self) -> float:
        """Simple SE estimate based on test information."""
        if not self._items_administered:
            return float("inf")

        theta_arr = np.array([[self._current_theta]])
        total_info = 0.0
        for item_idx in self._items_administered:
            info = self.model.information(theta_arr, item_idx=item_idx)
            total_info += float(info.sum())

        if total_info > 0:
            return float(1.0 / np.sqrt(total_info))
        return float("inf")

    def get_result(self) -> CATResult:
        """Get the final CAT result.

        Returns
        -------
        CATResult
            Complete result of the CAT session.

        Raises
        ------
        RuntimeError
            If the CAT session is not yet complete.
        """
        if not self._is_complete:
            raise RuntimeError("CAT session is not complete")

        return CATResult(
            theta=self._current_theta,
            standard_error=self._current_se,
            items_administered=list(self._items_administered),
            responses=np.array(self._responses, dtype=np.int_),
            n_items_administered=len(self._items_administered),
            stopping_reason=self._stopping_reason,
            theta_history=list(self._theta_history),
            se_history=list(self._se_history),
            item_info_history=list(self._info_history),
        )

    def run_simulation(
        self,
        true_theta: float,
        response_generator: Any | None = None,
    ) -> CATResult:
        """Simulate a complete CAT session with known true ability.

        Parameters
        ----------
        true_theta : float
            True ability level for response simulation.
        response_generator : callable | None, optional
            Custom response generator function. If None, uses
            probabilistic responses based on model probabilities.

        Returns
        -------
        CATResult
            Result of the simulated CAT session.
        """
        self.reset()

        while not self._is_complete:
            # Select next item
            item_idx = self.select_next_item()
            self._pending_item = item_idx

            # Generate response
            if response_generator is not None:
                response = response_generator(item_idx, true_theta)
            else:
                response = self._generate_response(item_idx, true_theta)

            # Administer item
            self.administer_item(response)

        return self.get_result()

    def _generate_response(self, item_idx: int, true_theta: float) -> int:
        """Generate a probabilistic response based on true ability.

        Parameters
        ----------
        item_idx : int
            Index of the item.
        true_theta : float
            True ability level.

        Returns
        -------
        int
            Generated response.
        """
        theta_arr = np.array([[true_theta]])
        prob = self.model.probability(theta_arr, item_idx=item_idx)

        if prob.ndim == 1 or (prob.ndim == 2 and prob.shape[1] == 1):
            # Dichotomous item
            p = float(prob.ravel()[0])
            return int(self.rng.random() < p)
        else:
            # Polytomous item - sample from category probabilities
            probs = prob.ravel()
            return int(self.rng.choice(len(probs), p=probs))

    def run_batch_simulation(
        self,
        true_thetas: NDArray[np.float64] | list[float],
        n_replications: int = 1,
        use_rust: bool = True,
    ) -> list[CATResult]:
        """Simulate CAT for multiple examinees.

        Parameters
        ----------
        true_thetas : array-like
            Array of true ability values.
        n_replications : int, optional
            Number of replications per theta value. Default is 1.
        use_rust : bool, optional
            Use Rust backend for parallel simulation if available. Default is True.
            Only works with 2PL models using MFI selection.

        Returns
        -------
        list[CATResult]
            List of CAT results for all simulations.
        """
        thetas = np.asarray(true_thetas).ravel()

        # Check if we can use Rust backend
        can_use_rust = use_rust and RUST_AVAILABLE and self._can_use_rust_simulation()

        if can_use_rust:
            return self._run_batch_rust(thetas, n_replications)

        # Python fallback
        results = []
        for theta in thetas:
            for _ in range(n_replications):
                result = self.run_simulation(float(theta))
                results.append(result)

        return results

    def _can_use_rust_simulation(self) -> bool:
        """Check if Rust simulation can be used."""
        # Only 2PL models with MFI selection

        if not isinstance(self._selection, MaxFisherInformation):
            return False

        # Check if model has discrimination and difficulty parameters
        params = self.model.parameters
        if "discrimination" not in params or "difficulty" not in params:
            return False

        # Only unidimensional
        if self.model.n_factors != 1:
            return False

        # Check discrimination shape (must be 1D)
        disc = params["discrimination"]
        if disc.ndim != 1:
            return False

        return True

    def _get_quadrature(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get quadrature points and weights."""
        from mirt.estimation.quadrature import GaussHermiteQuadrature

        quad = GaussHermiteQuadrature(self.n_quadpts, n_dimensions=1)
        return quad.nodes.ravel(), quad.weights.ravel()

    def _run_batch_rust(
        self,
        true_thetas: NDArray[np.float64],
        n_replications: int,
    ) -> list[CATResult]:
        """Run batch simulation using Rust backend."""
        params = self.model.parameters
        disc = params["discrimination"].astype(np.float64)
        diff = params["difficulty"].astype(np.float64)

        quad_points, quad_weights = self._get_quadrature()

        # Get stopping parameters
        se_threshold = 0.3  # Default
        max_items = self.model.n_items
        min_items = 1

        # Extract from stopping rule if possible
        if hasattr(self._stopping, "threshold"):
            se_threshold = self._stopping.threshold
        if hasattr(self._stopping, "max_items"):
            max_items = self._stopping.max_items
        if hasattr(self._stopping, "min_items"):
            min_items = self._stopping.min_items

        # Run Rust simulation
        seed = (
            self.seed
            if self.seed is not None
            else np.random.default_rng().integers(0, 2**31)
        )
        result = rust_cat_simulate_batch(
            true_thetas,
            disc,
            diff,
            quad_points,
            quad_weights,
            se_threshold,
            max_items,
            min_items,
            n_replications,
            seed,
        )

        if result is None:
            # Fallback to Python
            results = []
            for theta in true_thetas:
                for _ in range(n_replications):
                    res = self.run_simulation(float(theta))
                    results.append(res)
            return results

        theta_est, se_est, n_items, true_theta_out = result

        # Convert to CATResult objects
        results = []
        for i in range(len(theta_est)):
            results.append(
                CATResult(
                    theta=float(theta_est[i]),
                    standard_error=float(se_est[i]),
                    items_administered=[],  # Not tracked in Rust
                    responses=np.array([], dtype=np.int_),
                    n_items_administered=int(n_items[i]),
                    stopping_reason="SE threshold reached"
                    if se_est[i] <= se_threshold
                    else "Max items reached",
                    theta_history=[],
                    se_history=[],
                    item_info_history=[],
                )
            )

        return results

    def compute_conditional_mse(
        self,
        true_thetas: NDArray[np.float64] | list[float],
        n_replications: int = 100,
        use_rust: bool = True,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Compute conditional MSE at different ability levels.

        Parameters
        ----------
        true_thetas : array-like
            Array of true ability values to evaluate.
        n_replications : int, optional
            Number of replications per theta. Default is 100.
        use_rust : bool, optional
            Use Rust backend for parallel computation if available. Default is True.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, NDArray]
            Tuple of (thetas, biases, MSEs, avg_items).
        """
        thetas = np.asarray(true_thetas).ravel()

        # Check if we can use Rust backend
        can_use_rust = use_rust and RUST_AVAILABLE and self._can_use_rust_simulation()

        if can_use_rust:
            params = self.model.parameters
            disc = params["discrimination"].astype(np.float64)
            diff = params["difficulty"].astype(np.float64)

            quad_points, quad_weights = self._get_quadrature()

            # Get stopping parameters
            se_threshold = 0.3
            max_items = self.model.n_items
            min_items = 1

            if hasattr(self._stopping, "threshold"):
                se_threshold = self._stopping.threshold
            if hasattr(self._stopping, "max_items"):
                max_items = self._stopping.max_items
            if hasattr(self._stopping, "min_items"):
                min_items = self._stopping.min_items

            seed = (
                self.seed
                if self.seed is not None
                else np.random.default_rng().integers(0, 2**31)
            )
            result = rust_cat_conditional_mse(
                thetas,
                disc,
                diff,
                quad_points,
                quad_weights,
                se_threshold,
                max_items,
                min_items,
                n_replications,
                seed,
            )

            if result is not None:
                return result

        # Python fallback
        biases = np.zeros(len(thetas))
        mses = np.zeros(len(thetas))
        avg_items = np.zeros(len(thetas))

        for i, true_theta in enumerate(thetas):
            estimates = []
            n_items_list = []
            for _ in range(n_replications):
                result = self.run_simulation(float(true_theta))
                estimates.append(result.theta)
                n_items_list.append(result.n_items_administered)

            estimates = np.array(estimates)
            bias = np.mean(estimates) - true_theta
            mse = np.mean((estimates - true_theta) ** 2)

            biases[i] = bias
            mses[i] = mse
            avg_items[i] = np.mean(n_items_list)

        return thetas, biases, mses, avg_items

    def __repr__(self) -> str:
        status = "complete" if self._is_complete else "in_progress"
        return (
            f"CATEngine(n_items={self.model.n_items}, "
            f"administered={len(self._items_administered)}, "
            f"status={status})"
        )
