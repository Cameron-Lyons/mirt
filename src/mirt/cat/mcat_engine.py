"""MCAT engine for orchestrating multidimensional computerized adaptive testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.cat._engine_common import (
    build_administered_response_matrix,
    configure_content_constraint,
    configure_exposure_control,
    consume_pending_item,
    finalize_administered_item,
    get_pending_item,
    initialize_common_engine,
    is_numerical_scoring_failure,
    record_item_administration,
    reset_session_state,
    run_simulation_loop,
    score_administered_responses,
    select_constrained_item,
    simulate_error_moments,
    validate_replications,
    validate_simulation_values,
    validate_test_length,
    warn_scoring_failure,
)
from mirt.cat._names import canonical_name
from mirt.cat._stopping_common import item_limit
from mirt.cat.content import ContentConstraint
from mirt.cat.exposure import ExposureControl
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
    from mirt.results.fit_result import FitResult

_SCORING_METHODS = ("EAP", "MAP")
_STOPPING_NAMES = (
    "trace",
    "determinant",
    "max_se",
    "avg_se",
    "max_items",
    "theta_change",
    "classification",
    "combined",
)


def _square_matrix(value: Any, n_factors: int, name: str) -> NDArray[np.float64]:
    """Return a copied ``(n_factors, n_factors)`` float matrix."""
    matrix = np.array(value, dtype=np.float64, copy=True)
    if matrix.shape != (n_factors, n_factors):
        raise ValueError(
            f"{name} must have shape ({n_factors}, {n_factors}), got {matrix.shape}"
        )
    return matrix


def _named_stopping_rule(
    name: str, trace_threshold: float, max_items: int | None
) -> MCATStoppingRule:
    """Build a stopping rule named by a canonical ``MCATEngine`` option."""
    if name == "trace":
        return CovarianceTraceStop(trace_threshold)
    if name == "max_items":
        if max_items is None:
            raise ValueError("stopping_rule='max_items' requires max_items")
        return MaxItemsMCATStop(max_items)
    if name in ("classification", "combined"):
        raise ValueError(
            f"stopping_rule={name!r} needs arguments; pass an MCATStoppingRule instance"
        )
    return create_mcat_stopping_rule(name)


def _validate_batch_controls(
    true_thetas: NDArray[np.float64],
    n_replications: int,
    *,
    n_factors: int,
) -> tuple[NDArray[np.float64], int]:
    """Validate an entire diagnostic request before running any sessions."""
    thetas = validate_simulation_values(true_thetas, name="true_thetas")
    if thetas.ndim == 1:
        thetas = thetas.reshape(1, -1)
    if thetas.ndim != 2 or thetas.shape[1] != n_factors:
        raise ValueError(f"true_thetas must have shape (n_examinees, {n_factors})")
    if thetas.shape[0] == 0:
        raise ValueError("true_thetas must be non-empty")
    return np.ascontiguousarray(thetas), validate_replications(n_replications)


class MCATEngine:
    """Engine for multidimensional computerized adaptive testing.

    Orchestrates the adaptive testing process for multidimensional IRT models,
    including item selection, ability estimation across multiple dimensions,
    stopping rule evaluation, and constraint handling.

    Parameters
    ----------
    model : BaseItemModel | FitResult
        Fitted multidimensional IRT model containing item parameters, or the
        ``FitResult`` of its calibration, whose ``latent_mean`` and
        ``latent_covariance`` (for example the factor correlations) then
        supply the default scoring prior.
    item_selection : MCATSelectionStrategy | str
        Item selection strategy or name. Default is "D-optimality".
        Options: "D-optimality", "A-optimality", "C-optimality", "KL",
        "Bayesian", "random".
    stopping_rule : MCATStoppingRule | str
        Stopping rule or name. Default is "trace" (uses ``trace_threshold``).
        Options: "trace", "determinant", "max_se", "avg_se", "max_items"
        (uses ``max_items``, which is then required), and "theta_change".
        Rules that need further settings, such as
        ``CompositeClassificationStop``, are passed as objects. Progressive
        exposure control plans over the length the rules enforce:
        ``max_items`` or a ``MaxItemsMCATStop`` in the rule, otherwise the
        pool size.
    scoring_method : {"EAP", "MAP"}
        Method for ability estimation. Default is "EAP", which retains the
        full posterior covariance. MAP uses a diagonal SE approximation.
        Names ignore case. Both use the normal prior
        ``N(prior_mean, prior_cov)``.
    initial_theta : NDArray[np.float64] | None
        Ability estimates before the first response, which select the first
        item. Default is ``prior_mean``.
    initial_covariance : NDArray[np.float64] | None
        Covariance before the first response. With no response it is the
        prior covariance, so it is the default scoring ``prior_cov`` when
        given. Default is ``prior_cov``.
    trace_threshold : float
        Trace threshold for stopping rule. Default is 0.5.
    max_items : int | None
        Maximum number of items to administer. Default is None.
    min_items : int
        Minimum number of items before stopping, a non-negative integer that
        may not exceed ``max_items`` or the length a maximum-length stopping
        rule allows. Default is 1.
    exposure_control : ExposureControl | str | None
        Exposure control method. Default is None (no control).
    content_constraint : ContentConstraint | None
        Content balancing constraint. Default is None.
    n_quadpts : int
        Number of quadrature points per factor for EAP scoring, an integer
        of at least 5. Default is 21.
    theta_bounds : tuple[float, float]
        Bounds for MAP ability estimation. Default is (-4.0, 4.0).
    seed : int | None
        Random seed for reproducibility.
    prior_mean : NDArray[np.float64] | None
        Mean of the normal scoring prior. Default is the ``latent_mean`` of
        a ``FitResult`` that has one, otherwise zeros.
    prior_cov : NDArray[np.float64] | None
        Positive definite covariance of the normal scoring prior, used by
        every ability update. Default is ``initial_covariance`` when given,
        else the ``latent_covariance`` of a ``FitResult`` that has one,
        otherwise the identity.

    Raises
    ------
    ValueError
        If the model is not fitted or is unidimensional, or if
        ``scoring_method``, ``n_quadpts``, ``theta_bounds``, the prior, the
        initial state, or ``min_items`` is invalid.

    Examples
    --------
    >>> from mirt import fit_mirt
    >>> from mirt.cat import MCATEngine
    >>> result = fit_mirt(data, model="2PL", n_factors=2)
    >>> mcat = MCATEngine(result, trace_threshold=0.5, max_items=30)
    >>> state = mcat.get_current_state()
    >>> while not state.is_complete:
    ...     response = get_response(state.next_item)
    ...     state = mcat.administer_item(response)
    >>> print(mcat.get_result().summary())
    """

    def __init__(
        self,
        model: BaseItemModel | FitResult,
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
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
    ):
        from mirt.results._common import resolve_latent_prior
        from mirt.scoring._common import resolve_prior_distribution

        model, fitted_mean, fitted_cov = resolve_latent_prior(model)
        initialize_common_engine(
            self,
            model=model,
            scoring_method=scoring_method,
            scoring_methods=_SCORING_METHODS,
            n_quadpts=n_quadpts,
            theta_bounds=theta_bounds,
            seed=seed,
            engine_name="MCAT",
        )

        if model.n_factors < 2:
            raise ValueError(
                f"MCATEngine requires a multidimensional model with n_factors >= 2, "
                f"got n_factors={model.n_factors}. Use CATEngine for unidimensional models."
            )

        self.n_factors = model.n_factors

        if initial_covariance is not None:
            initial_covariance = _square_matrix(
                initial_covariance, self.n_factors, "initial_covariance"
            )
        if prior_mean is None:
            prior_mean = fitted_mean
        if prior_cov is None:
            prior_cov = (
                initial_covariance if initial_covariance is not None else fitted_cov
            )
        mean, cov = resolve_prior_distribution(
            n_factors=self.n_factors, prior_mean=prior_mean, prior_cov=prior_cov
        )
        self.prior_mean: NDArray[np.float64] = mean
        self.prior_cov: NDArray[np.float64] = cov

        if initial_theta is None:
            self.initial_theta = self.prior_mean.copy()
        else:
            self.initial_theta = np.asarray(initial_theta)
            if self.initial_theta.shape != (self.n_factors,):
                raise ValueError(
                    f"initial_theta must have shape ({self.n_factors},), "
                    f"got {self.initial_theta.shape}"
                )

        self.initial_covariance = (
            self.prior_cov.copy() if initial_covariance is None else initial_covariance
        )

        if isinstance(item_selection, str):
            self._selection = create_mcat_selection_strategy(item_selection)
        else:
            self._selection = item_selection

        rule_name = None
        if isinstance(stopping_rule, str):
            rule_name = canonical_name(
                stopping_rule, _STOPPING_NAMES, kind="MCAT stopping rule"
            )
            base_rule = _named_stopping_rule(rule_name, trace_threshold, max_items)
        else:
            base_rule = stopping_rule

        rules: list[MCATStoppingRule] = [base_rule]
        # A rule named "max_items" already enforces max_items.
        if max_items is not None and rule_name != "max_items":
            rules.append(MaxItemsMCATStop(max_items))
        min_items = validate_test_length(
            min_items,
            rules,
            length_rule=MaxItemsMCATStop,
            combined_rule=CombinedMCATStop,
        )

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

    @property
    def _selection_horizon(self) -> int:
        """Return the test length that selection and exposure plan over.

        Strata, shadow tests and progressive exposure span the length the
        stopping rules enforce, whether set by ``max_items`` or a rule. It is
        read at each selection, so a rule changed after construction counts.
        """
        n_items = self.model.n_items
        limit = item_limit(self._stopping, MaxItemsMCATStop, CombinedMCATStop)
        return n_items if limit is None else min(limit, n_items)

    def reset(self) -> None:
        """Reset the engine for a new examinee.

        Resetting an unused session reuses its exposure count. Once item
        selection begins, a reset opens and counts a new exposure session.
        """
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
            next_item = get_pending_item(self)

        return self._build_state(next_item)

    def _build_state(self, next_item: int | None) -> MCATState:
        """Create a state snapshot with an explicitly resolved next item."""
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

        return get_pending_item(self)

    def _select_next_item(self) -> int:
        """Internal item selection with constraint handling."""
        strategy = self._selection
        return select_constrained_item(
            self,
            criteria=lambda eligible: strategy.get_item_criteria(
                self.model,
                self._current_theta,
                self._current_covariance,
                eligible,
            ),
            select=lambda eligible: strategy.select_item(
                self.model,
                self._current_theta,
                self._current_covariance,
                eligible,
                self._items_administered,
                self._responses,
            ),
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
        ValueError
            If the response is not an integer-valued numeric scalar in the
            selected item's category range. The selected item remains pending
            so a corrected response can be submitted.
        """
        if self._is_complete:
            raise RuntimeError("MCAT session is already complete")

        item_idx, response = consume_pending_item(self, response)

        theta_arr = self._current_theta.reshape(1, -1)
        record_item_administration(
            self,
            item_idx=item_idx,
            response=response,
            theta_arr=theta_arr,
        )

        self._update_theta()

        self._theta_history.append(self._current_theta.copy())
        self._se_history.append(self.current_standard_error.copy())
        self._covariance_history.append(self._current_covariance.copy())

        finalize_administered_item(self, self._build_state(next_item=None))

        return self.get_current_state()

    def _scoring_prior(
        self,
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        """Return the scoring prior's mean and covariance for the scorers.

        Each part is None when it takes its standard-normal default.
        """
        mean = np.asarray(self.prior_mean, dtype=np.float64)
        cov = np.asarray(self.prior_cov, dtype=np.float64)
        return (
            None if not np.any(mean) else mean,
            None if np.array_equal(cov, np.eye(self.n_factors)) else cov,
        )

    def _update_theta(self) -> None:
        """Update ability estimates based on administered items.

        A numerical scoring failure, such as a posterior without finite mass,
        warns and keeps the previous estimate. Configuration errors
        propagate.
        """
        try:
            if self.scoring_method == "EAP":
                from mirt.cat._eap import score_binary_eap
                from mirt.scoring import ability_posterior

                moments = score_binary_eap(self)
                if moments is not None:
                    theta, covariance = moments
                else:
                    prior_mean, prior_cov = self._scoring_prior()
                    posterior = ability_posterior(
                        self.model,
                        build_administered_response_matrix(self),
                        n_quadpts=self.n_quadpts,
                        prior_mean=prior_mean,
                        prior_cov=prior_cov,
                    )
                    theta = posterior.mean.ravel()
                    centered_points = posterior.points - theta
                    covariance = centered_points.T @ (
                        posterior.weights[0, :, None] * centered_points
                    )
            else:
                result = score_administered_responses(self, bounds=self.theta_bounds)
                theta = result.theta.ravel()
                covariance = np.diag(result.standard_error.ravel() ** 2)

            self._current_theta = theta
            self._current_covariance = covariance

        except (ArithmeticError, np.linalg.LinAlgError, ValueError) as error:
            if not is_numerical_scoring_failure(error):
                raise
            warn_scoring_failure(self, error, "keeping the previous estimate")

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
            polytomous=bool(self.model.is_polytomous),
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
        true_theta = validate_simulation_values(true_theta, name="true_theta")
        if true_theta.shape != (self.n_factors,):
            raise ValueError(
                f"true_theta must have shape ({self.n_factors},), "
                f"got {true_theta.shape}"
            )

        return run_simulation_loop(
            self,
            true_theta,
            response_generator=response_generator,
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
        true_thetas, n_replications = _validate_batch_controls(
            true_thetas, n_replications, n_factors=self.n_factors
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
        true_thetas, n_replications = _validate_batch_controls(
            true_thetas, n_replications, n_factors=self.n_factors
        )

        n_points = true_thetas.shape[0]
        bias = np.zeros((n_points, self.n_factors))
        mse = np.zeros((n_points, self.n_factors))
        avg_items = np.zeros(n_points)

        for i, true_theta in enumerate(true_thetas):
            bias[i], mse[i], avg_items[i] = simulate_error_moments(
                self, true_theta, n_replications=n_replications
            )

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
