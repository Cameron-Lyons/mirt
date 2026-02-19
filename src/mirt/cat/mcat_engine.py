"""MCAT engine for orchestrating multidimensional computerized adaptive testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.cat._engine_common import (
    configure_content_constraint,
    configure_exposure_control,
    consume_pending_item,
    finalize_administered_item,
    reset_session_state,
    run_simulation_loop,
)
from mirt.cat.content import ContentConstraint
from mirt.cat.exposure import (
    ExposureControl,
)
from mirt.cat.mcat_selection import (
    MCATSelectionStrategy,
    create_mcat_selection_strategy,
)
from mirt.cat.mcat_stopping import (
    CombinedMCATStop,
    CovarianceTraceStop,
    MaxItemsMCATStop,
    MCATStoppingRule,
    create_mcat_stopping_rule,
)
from mirt.cat.results import MCATResult, MCATState
from mirt.constants import PROB_CLIP_MAX, PROB_CLIP_MIN

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class MCATEngine:
    """Engine for multidimensional computerized adaptive testing.

    Orchestrates the adaptive testing process for multidimensional IRT models,
    including item selection, ability estimation across multiple dimensions,
    stopping rule evaluation, and constraint handling.

    Parameters
    ----------
    model : BaseItemModel
        Fitted multidimensional IRT model containing item parameters.
    item_selection : MCATSelectionStrategy | str
        Item selection strategy or name. Default is "D-optimality".
        Options: "D-optimality", "A-optimality", "C-optimality", "KL",
        "Bayesian", "random".
    stopping_rule : MCATStoppingRule | str
        Stopping rule or name. Default is "trace".
        Options: "trace", "determinant", "max_se", "avg_se", "max_items",
        "theta_change".
    scoring_method : {"EAP", "MAP"}
        Method for ability estimation. Default is "EAP".
    initial_theta : NDArray[np.float64] | None
        Initial ability estimates. Default is zeros.
    initial_covariance : NDArray[np.float64] | None
        Initial covariance matrix. Default is identity.
    trace_threshold : float
        Trace threshold for stopping rule. Default is 0.5.
    max_items : int | None
        Maximum number of items to administer. Default is None.
    min_items : int
        Minimum number of items before stopping. Default is 1.
    exposure_control : ExposureControl | str | None
        Exposure control method. Default is None (no control).
    content_constraint : ContentConstraint | None
        Content balancing constraint. Default is None.
    n_quadpts : int
        Number of quadrature points for EAP scoring. Default is 21.
    theta_bounds : tuple[float, float]
        Bounds for ability estimation. Default is (-4.0, 4.0).
    seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> from mirt import fit_mirt
    >>> from mirt.cat import MCATEngine
    >>> result = fit_mirt(data, model="MIRT", n_factors=2)
    >>> mcat = MCATEngine(result.model, trace_threshold=0.5, max_items=30)
    >>> state = mcat.get_current_state()
    >>> while not state.is_complete:
    ...     response = get_response(state.next_item)
    ...     state = mcat.administer_item(response)
    >>> print(mcat.get_result().summary())
    """

    def __init__(
        self,
        model: BaseItemModel,
        item_selection: MCATSelectionStrategy | str = "D-optimality",
        stopping_rule: MCATStoppingRule | str = "trace",
        scoring_method: Literal["EAP", "MAP"] = "EAP",
        initial_theta: NDArray[np.float64] | None = None,
        initial_covariance: NDArray[np.float64] | None = None,
        trace_threshold: float = 0.5,
        max_items: int | None = None,
        min_items: int = 1,
        exposure_control: ExposureControl | str | None = None,
        content_constraint: ContentConstraint | None = None,
        n_quadpts: int = 21,
        theta_bounds: tuple[float, float] = (-4.0, 4.0),
        seed: int | None = None,
    ):
        if not model.is_fitted:
            raise ValueError("Model must be fitted before use in MCAT")

        if model.n_factors < 2:
            raise ValueError(
                f"MCATEngine requires a multidimensional model with n_factors >= 2, "
                f"got n_factors={model.n_factors}. Use CATEngine for unidimensional models."
            )

        self.model = model
        self.n_factors = model.n_factors
        self.scoring_method = scoring_method
        self.n_quadpts = n_quadpts
        self.theta_bounds = theta_bounds
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if initial_theta is None:
            self.initial_theta = np.zeros(self.n_factors)
        else:
            self.initial_theta = np.asarray(initial_theta)
            if self.initial_theta.shape != (self.n_factors,):
                raise ValueError(
                    f"initial_theta must have shape ({self.n_factors},), "
                    f"got {self.initial_theta.shape}"
                )

        if initial_covariance is None:
            self.initial_covariance = np.eye(self.n_factors)
        else:
            self.initial_covariance = np.asarray(initial_covariance)
            if self.initial_covariance.shape != (self.n_factors, self.n_factors):
                raise ValueError(
                    f"initial_covariance must have shape "
                    f"({self.n_factors}, {self.n_factors}), "
                    f"got {self.initial_covariance.shape}"
                )

        if isinstance(item_selection, str):
            self._selection = create_mcat_selection_strategy(item_selection)
        else:
            self._selection = item_selection

        if isinstance(stopping_rule, str):
            if stopping_rule == "trace":
                base_rule = CovarianceTraceStop(trace_threshold)
            else:
                base_rule = create_mcat_stopping_rule(stopping_rule)
        else:
            base_rule = stopping_rule

        rules: list[MCATStoppingRule] = [base_rule]
        if max_items is not None:
            rules.append(MaxItemsMCATStop(max_items))

        if len(rules) > 1 or min_items > 1:
            self._stopping = CombinedMCATStop(rules, operator="or", min_items=min_items)
        else:
            self._stopping = base_rule

        self._exposure = configure_exposure_control(exposure_control, seed=seed)
        self._content = configure_content_constraint(content_constraint)

        self._current_theta = self.initial_theta.copy()
        self._current_covariance = self.initial_covariance.copy()
        self._items_administered: list[int] = []
        self._responses: list[int] = []
        self._available_items: set[int] = set(range(model.n_items))
        self._theta_history: list[NDArray[np.float64]] = []
        self._se_history: list[NDArray[np.float64]] = []
        self._covariance_history: list[NDArray[np.float64]] = []
        self._info_history: list[float] = []
        self._is_complete = False
        self._stopping_reason = ""

    def reset(self) -> None:
        """Reset the engine for a new examinee."""
        self._current_theta = self.initial_theta.copy()
        self._current_covariance = self.initial_covariance.copy()
        reset_session_state(
            self,
            n_items=self.model.n_items,
            history_attrs=(
                "_theta_history",
                "_se_history",
                "_covariance_history",
                "_info_history",
            ),
        )

    @property
    def current_standard_error(self) -> NDArray[np.float64]:
        """Current standard errors (sqrt of diagonal of covariance)."""
        return np.sqrt(np.diag(self._current_covariance))

    def get_current_state(self) -> MCATState:
        """Get the current state of the MCAT session.

        Returns
        -------
        MCATState
            Current state including theta vector, covariance, items administered.
        """
        next_item = None
        if not self._is_complete and self._available_items:
            next_item = self._select_next_item()

        return MCATState(
            theta=self._current_theta.copy(),
            covariance=self._current_covariance.copy(),
            standard_error=self.current_standard_error,
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
            If the MCAT session is complete or no items available.
        """
        if self._is_complete:
            raise RuntimeError("MCAT session is complete")
        if not self._available_items:
            raise RuntimeError("No items available for selection")

        return self._select_next_item()

    def _select_next_item(self) -> int:
        """Internal item selection with constraint handling."""
        content_eligible = self._content.filter_items(
            self._available_items, self._items_administered
        )

        exposure_eligible = self._exposure.filter_items(
            content_eligible, self.model, float(self._current_theta[0])
        )

        return self._selection.select_item(
            self.model,
            self._current_theta,
            self._current_covariance,
            exposure_eligible,
            self._items_administered,
            self._responses,
        )

    def administer_item(self, response: int) -> MCATState:
        """Record a response and update the MCAT state.

        Parameters
        ----------
        response : int
            Response to the administered item (0/1 for dichotomous,
            0..k for polytomous).

        Returns
        -------
        MCATState
            Updated MCAT state after processing the response.

        Raises
        ------
        RuntimeError
            If the MCAT session is already complete.
        """
        if self._is_complete:
            raise RuntimeError("MCAT session is already complete")

        item_idx = consume_pending_item(self)

        theta_arr = self._current_theta.reshape(1, -1)
        item_info = float(self.model.information(theta_arr, item_idx=item_idx).sum())
        self._info_history.append(item_info)

        self._items_administered.append(item_idx)
        self._responses.append(response)
        self._available_items.discard(item_idx)

        self._exposure.update(item_idx)

        self._update_theta()

        self._theta_history.append(self._current_theta.copy())
        self._se_history.append(self.current_standard_error.copy())
        self._covariance_history.append(self._current_covariance.copy())

        state = self.get_current_state()
        finalize_administered_item(self, state)

        return self.get_current_state()

    def _update_theta(self) -> None:
        """Update ability estimates based on administered items."""
        from mirt.scoring import fscores

        responses = np.full((1, self.model.n_items), -1, dtype=np.int_)
        for item_idx, resp in zip(self._items_administered, self._responses):
            responses[0, item_idx] = resp

        try:
            result = fscores(
                self.model,
                responses,
                method=self.scoring_method,
                n_quadpts=self.n_quadpts,
            )
            theta = result.theta
            se = result.standard_error

            if theta.ndim == 1:
                self._current_theta = theta
            else:
                self._current_theta = theta.ravel()

            if se.ndim == 1:
                se_arr = se
            else:
                se_arr = se.ravel()

            self._current_covariance = np.diag(se_arr**2)

        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ):
            pass

    def get_result(self) -> MCATResult:
        """Get the final MCAT result.

        Returns
        -------
        MCATResult
            Complete result of the MCAT session.

        Raises
        ------
        RuntimeError
            If the MCAT session is not yet complete.
        """
        if not self._is_complete:
            raise RuntimeError("MCAT session is not complete")

        return MCATResult(
            theta=self._current_theta.copy(),
            covariance=self._current_covariance.copy(),
            standard_error=self.current_standard_error.copy(),
            items_administered=list(self._items_administered),
            responses=np.array(self._responses, dtype=np.int_),
            n_items_administered=len(self._items_administered),
            stopping_reason=self._stopping_reason,
            theta_history=list(self._theta_history),
            se_history=list(self._se_history),
            covariance_history=list(self._covariance_history),
            item_info_history=list(self._info_history),
        )

    def run_simulation(
        self,
        true_theta: NDArray[np.float64],
        response_generator: Any | None = None,
    ) -> MCATResult:
        """Simulate a complete MCAT session with known true abilities.

        Parameters
        ----------
        true_theta : NDArray[np.float64]
            True ability levels, shape (n_factors,).
        response_generator : callable | None
            Custom response generator function. If None, uses
            probabilistic responses based on model probabilities.

        Returns
        -------
        MCATResult
            Result of the simulated MCAT session.
        """
        self.reset()

        true_theta = np.asarray(true_theta)
        if true_theta.shape != (self.n_factors,):
            raise ValueError(
                f"true_theta must have shape ({self.n_factors},), "
                f"got {true_theta.shape}"
            )

        return run_simulation_loop(
            self,
            true_theta,
            response_generator=response_generator,
            reset=False,
        )

    def _generate_response(self, item_idx: int, true_theta: NDArray[np.float64]) -> int:
        """Generate a probabilistic response based on true abilities.

        Parameters
        ----------
        item_idx : int
            Index of the item.
        true_theta : NDArray[np.float64]
            True ability levels.

        Returns
        -------
        int
            Generated response.
        """
        theta_arr = true_theta.reshape(1, -1)
        prob = self.model.probability(theta_arr, item_idx=item_idx)

        if prob.ndim == 1 or (prob.ndim == 2 and prob.shape[1] == 1):
            p = float(prob.ravel()[0])
            p = np.clip(p, PROB_CLIP_MIN, PROB_CLIP_MAX)
            return int(self.rng.random() < p)
        else:
            probs = prob.ravel()
            probs = np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
            probs = probs / probs.sum()
            return int(self.rng.choice(len(probs), p=probs))

    def run_batch_simulation(
        self,
        true_thetas: NDArray[np.float64],
        n_replications: int = 1,
    ) -> list[MCATResult]:
        """Simulate MCAT for multiple examinees.

        Parameters
        ----------
        true_thetas : NDArray[np.float64]
            Array of true ability values, shape (n_examinees, n_factors).
        n_replications : int
            Number of replications per theta vector. Default is 1.

        Returns
        -------
        list[MCATResult]
            List of MCAT results for all simulations.
        """
        true_thetas = np.asarray(true_thetas)
        if true_thetas.ndim == 1:
            true_thetas = true_thetas.reshape(1, -1)

        if true_thetas.shape[1] != self.n_factors:
            raise ValueError(
                f"true_thetas must have {self.n_factors} columns, "
                f"got {true_thetas.shape[1]}"
            )

        results = []
        for theta in true_thetas:
            for _ in range(n_replications):
                result = self.run_simulation(theta)
                results.append(result)

        return results

    def compute_conditional_metrics(
        self,
        true_thetas: NDArray[np.float64],
        n_replications: int = 100,
    ) -> dict[str, NDArray[np.float64]]:
        """Compute conditional bias and MSE at different ability levels.

        Parameters
        ----------
        true_thetas : NDArray[np.float64]
            Array of true ability vectors, shape (n_points, n_factors).
        n_replications : int
            Number of replications per theta vector. Default is 100.

        Returns
        -------
        dict[str, NDArray[np.float64]]
            Dictionary containing:
            - 'true_thetas': The input true theta values
            - 'bias': Bias for each dimension at each theta point
            - 'mse': MSE for each dimension at each theta point
            - 'avg_items': Average number of items at each theta point
        """
        true_thetas = np.asarray(true_thetas)
        if true_thetas.ndim == 1:
            true_thetas = true_thetas.reshape(1, -1)

        n_points = true_thetas.shape[0]
        bias = np.zeros((n_points, self.n_factors))
        mse = np.zeros((n_points, self.n_factors))
        avg_items = np.zeros(n_points)

        for i, true_theta in enumerate(true_thetas):
            estimates = []
            n_items_list = []

            for _ in range(n_replications):
                result = self.run_simulation(true_theta)
                estimates.append(result.theta)
                n_items_list.append(result.n_items_administered)

            estimates = np.array(estimates)
            bias[i] = np.mean(estimates, axis=0) - true_theta
            mse[i] = np.mean((estimates - true_theta) ** 2, axis=0)
            avg_items[i] = np.mean(n_items_list)

        return {
            "true_thetas": true_thetas,
            "bias": bias,
            "mse": mse,
            "avg_items": avg_items,
        }

    def __repr__(self) -> str:
        status = "complete" if self._is_complete else "in_progress"
        return (
            f"MCATEngine(n_items={self.model.n_items}, "
            f"n_factors={self.n_factors}, "
            f"administered={len(self._items_administered)}, "
            f"status={status})"
        )
