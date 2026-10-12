"""CAT engine for orchestrating computerized adaptive testing."""

from __future__ import annotations

import math
from collections.abc import Iterator
from copy import deepcopy
from functools import cache
from inspect import signature
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from mirt._backend_config import should_use_rust
from mirt._model_defaults import uses_builtin_model_hooks, uses_original_model_hook
from mirt.backends.rust.cat import cat_conditional_mse as rust_cat_conditional_mse
from mirt.backends.rust.cat import (
    cat_simulate_batch_full as rust_cat_simulate_batch_full,
)
from mirt.cat._engine_common import (
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
from mirt.cat._lockstep import conditional_error_moments as lockstep_error_moments
from mirt.cat._lockstep import simulate as simulate_lockstep
from mirt.cat._lockstep import supports_lockstep
from mirt.cat._names import canonical_name
from mirt.cat._native import register_native_defaults as _register_native_defaults
from mirt.cat._native import uses_native_defaults
from mirt.cat._stopping_common import finite_real, item_limit, positive_real
from mirt.cat.content import ContentConstraint, NoContentConstraint
from mirt.cat.exposure import (
    ExposureControl,
    NoExposureControl,
    ProgressiveRestricted,
)
from mirt.cat.results import CATResult, CATState
from mirt.cat.selection import (
    AStratified,
    ItemSelectionStrategy,
    MaxFisherInformation,
    _PosteriorGridSelection,
    _selection_strategy_class,
    create_selection_strategy,
)
from mirt.cat.shadow import ShadowTestSelection
from mirt.cat.stopping import (
    CombinedStop,
    MaxItemsStop,
    StandardErrorStop,
    StoppingRule,
    create_stopping_rule,
)
from mirt.constants import PROB_CLIP_MAX, PROB_CLIP_MIN

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

# Bound the (session x position) paths of one native batch call, and the
# sessions of one native conditional-MSE call.
_NATIVE_PATH_VALUES = 1 << 21
_NATIVE_MSE_SESSIONS = 1 << 18

_SCORING_METHODS = ("EAP", "MAP", "ML", "WLE")
_STOPPING_NAMES = (
    "SE",
    "max_items",
    "theta_change",
    "se_change",
    "classification",
    "min_items",
    "combined",
)


def _hook_accepts_test_length(hook: Any) -> bool:
    """Return whether a selection hook accepts a ``test_length`` keyword."""
    try:
        parameters = signature(hook).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "test_length" or parameter.kind is parameter.VAR_KEYWORD
        for parameter in parameters
    )


_cached_hook_accepts_test_length = cache(_hook_accepts_test_length)


def _accepts_test_length(strategy: ItemSelectionStrategy) -> bool:
    """Return whether horizon-aware selection hooks accept ``test_length``.

    Subclasses or instances that replace a hook with the original signature
    keep working; they schedule strata over their own ``test_length`` or the
    pool size.
    """
    for name in ("select_item", "get_item_criteria"):
        hook = getattr(strategy, name)
        hook = getattr(hook, "__func__", hook)
        try:
            accepts = _cached_hook_accepts_test_length(hook)
        except TypeError:  # unhashable callable
            accepts = _hook_accepts_test_length(hook)
        if not accepts:
            return False
    return True


def _named_stopping_rule(
    name: str, se_threshold: float, max_items: int | None
) -> StoppingRule:
    """Build a stopping rule named by a canonical ``CATEngine`` option."""
    if name == "SE":
        return StandardErrorStop(se_threshold)
    if name == "max_items":
        if max_items is None:
            raise ValueError("stopping_rule='max_items' requires max_items")
        return MaxItemsStop(max_items)
    if name == "classification":
        raise ValueError(
            "stopping_rule='classification' needs a cut score; pass "
            "ClassificationStop(cut_score, confidence) instead"
        )
    if name in ("min_items", "combined"):
        raise ValueError(
            f"stopping_rule={name!r} needs arguments; pass a StoppingRule instance"
        )
    return create_stopping_rule(name)


def _validate_batch_controls(
    true_thetas: NDArray[np.float64] | list[float],
    n_replications: int,
    use_rust: bool,
    vectorized: bool,
) -> tuple[NDArray[np.float64], int, bool, bool]:
    """Validate controls shared by unidimensional batch diagnostics."""
    thetas = validate_simulation_values(true_thetas, name="true_thetas").ravel()
    if thetas.size == 0:
        raise ValueError("true_thetas must be non-empty")
    n_replications = validate_replications(n_replications)
    if not isinstance(use_rust, (bool, np.bool_)):
        raise ValueError("use_rust must be boolean")
    if not isinstance(vectorized, (bool, np.bool_)):
        raise ValueError("vectorized must be boolean")
    return (
        np.ascontiguousarray(thetas),
        n_replications,
        bool(use_rust),
        bool(vectorized),
    )


@_register_native_defaults
class CATEngine:
    """Engine for computerized adaptive testing.

    Orchestrates the adaptive testing process including item selection,
    ability estimation, stopping rule evaluation, and constraint handling.

    Parameters
    ----------
    model : BaseItemModel | FitResult
        Fitted unidimensional IRT model containing item parameters, or the
        ``FitResult`` of its calibration, whose ``latent_mean`` and
        ``latent_covariance`` then supply the default scoring prior. Use
        :class:`~mirt.cat.mcat_engine.MCATEngine` for multidimensional
        models.
    item_selection : ItemSelectionStrategy | str, optional
        Item selection strategy or name. Default is "MFI".
        Options: "MFI", "MEI", "MPWI", "MLWI", "KL", "KLP", "MEPV", "Urry",
        "random", "a-stratified". The posterior-weighted criteria (MEI,
        MPWI, MLWI, KLP, MEPV) use the engine's ``n_quadpts`` and
        ``theta_bounds``.
    stopping_rule : StoppingRule | str, optional
        Stopping rule or name. Default is "SE".
        Options: "SE" (uses ``se_threshold``), "max_items" (uses
        ``max_items``, which is then required), "theta_change", and
        "se_change". Rules that need further settings, such as
        ``ClassificationStop(cut_score, confidence)``, are passed as objects.
        Selection that plans over the test length (a-stratified strata,
        shadow tests, progressive exposure) uses the length the rules
        enforce: ``max_items`` or a ``MaxItemsStop`` in the rule, otherwise
        the pool size.
    scoring_method : {"EAP", "MAP", "ML", "WLE"}, optional
        Method for ability estimation. Default is "EAP". Names ignore case.
        EAP and MAP use the normal prior ``N(prior_mean, prior_sd**2)``. ML
        has no finite estimate while every response is in the lowest
        category, or every response in the highest; following
        :func:`~mirt.fscores`, the estimate is then the corresponding
        ``theta_bounds`` limit with an infinite standard error, so SE-based
        rules wait for a mixed response pattern.
    initial_theta : float | None, optional
        Ability estimate before the first response, which selects the first
        item. Default is ``prior_mean``.
    se_threshold : float, optional
        Standard error threshold for SE stopping rule. Default is 0.3.
    max_items : int | None, optional
        Maximum number of items to administer. Default is None.
    min_items : int, optional
        Minimum number of items before stopping, a non-negative integer that
        may not exceed ``max_items`` or the length a maximum-length stopping
        rule allows. Default is 1.
    exposure_control : ExposureControl | str | None, optional
        Exposure control method. Default is None (no control).
    content_constraint : ContentConstraint | None, optional
        Content balancing constraint. Default is None (no constraint).
    n_quadpts : int, optional
        Number of quadrature points for EAP scoring, an integer of at least
        5. Default is 21.
    theta_bounds : tuple[float, float], optional
        Bounds for MAP, ML, and WLE ability estimation. Default is
        (-4.0, 4.0).
    seed : int | None, optional
        Random seed for reproducibility.
    prior_mean : float | None, optional
        Mean of the normal scoring prior. Default is the ``latent_mean`` of
        a ``FitResult`` that has one, otherwise 0.
    prior_sd : float | None, optional
        Positive standard deviation of the normal scoring prior. Default is
        the square root of a ``FitResult``'s ``latent_covariance``, otherwise
        1.

    Raises
    ------
    ValueError
        If the model is not fitted or is multidimensional, if
        ``scoring_method``, ``n_quadpts``, ``theta_bounds``, the prior, or
        ``min_items`` is invalid, or if ``min_items`` exceeds the maximum
        test length.

    Examples
    --------
    >>> from mirt import fit_mirt
    >>> from mirt.cat import CATEngine
    >>> result = fit_mirt(data, model="2PL")
    >>> cat = CATEngine(result, se_threshold=0.3, max_items=20)
    >>> state = cat.get_current_state()
    >>> while not state.is_complete:
    ...     response = get_response(state.next_item)  # User provides response
    ...     state = cat.administer_item(response)
    >>> print(f"Final theta: {state.theta:.2f}")
    """

    def __init__(
        self,
        model: BaseItemModel | FitResult,
        item_selection: ItemSelectionStrategy | str = "MFI",
        stopping_rule: StoppingRule | str = "SE",
        scoring_method: Literal["EAP", "MAP", "ML", "WLE"] = "EAP",
        initial_theta: float | None = None,
        se_threshold: float = 0.3,
        max_items: int | None = None,
        min_items: int = 1,
        exposure_control: ExposureControl | str | None = None,
        content_constraint: ContentConstraint | None = None,
        n_quadpts: int = 21,
        theta_bounds: tuple[float, float] = (-4.0, 4.0),
        seed: int | None = None,
        prior_mean: float | None = None,
        prior_sd: float | None = None,
    ):
        from mirt.results._common import resolve_latent_prior

        model, fitted_mean, fitted_cov = resolve_latent_prior(model)
        initialize_common_engine(
            self,
            model=model,
            scoring_method=scoring_method,
            scoring_methods=_SCORING_METHODS,
            n_quadpts=n_quadpts,
            theta_bounds=theta_bounds,
            seed=seed,
            engine_name="CAT",
        )
        if model.n_factors != 1:
            raise ValueError(
                f"CATEngine requires a unidimensional model, got "
                f"n_factors={model.n_factors}. Use MCATEngine for "
                "multidimensional models."
            )

        if prior_mean is None:
            prior_mean = 0.0 if fitted_mean is None else float(fitted_mean.ravel()[0])
        if prior_sd is None:
            prior_sd = (
                1.0 if fitted_cov is None else math.sqrt(float(fitted_cov.ravel()[0]))
            )
        self.prior_mean: float = finite_real(prior_mean, "prior_mean")
        self.prior_sd: float = positive_real(prior_sd, "prior_sd")
        self.initial_theta = self.prior_mean if initial_theta is None else initial_theta

        if isinstance(item_selection, str):
            selection_kwargs: dict[str, Any] = {}
            # Posterior-grid criteria (MEI, MPWI, MLWI, KLP, MEPV) integrate
            # over the engine's quadrature settings.
            if issubclass(
                _selection_strategy_class(item_selection), _PosteriorGridSelection
            ):
                selection_kwargs = {
                    "n_quadpts": n_quadpts,
                    "theta_bounds": theta_bounds,
                }
            self._selection = create_selection_strategy(
                item_selection, **selection_kwargs
            )
        else:
            self._selection = item_selection

        rule_name = None
        if isinstance(stopping_rule, str):
            rule_name = canonical_name(
                stopping_rule, _STOPPING_NAMES, kind="stopping rule"
            )
            base_rule = _named_stopping_rule(rule_name, se_threshold, max_items)
        else:
            base_rule = stopping_rule

        rules = [base_rule]
        # A rule named "max_items" already enforces max_items.
        if max_items is not None and rule_name != "max_items":
            rules.append(MaxItemsStop(max_items))
        min_items = validate_test_length(
            min_items, rules, length_rule=MaxItemsStop, combined_rule=CombinedStop
        )

        if len(rules) > 1 or min_items > 1:
            self._stopping = CombinedStop(rules, operator="or", min_items=min_items)
        else:
            self._stopping = base_rule

        self._exposure = configure_exposure_control(exposure_control, seed=seed)
        self._content = configure_content_constraint(content_constraint)
        if isinstance(self._selection, ShadowTestSelection) and isinstance(
            self._exposure, ProgressiveRestricted
        ):
            raise ValueError(
                "ShadowTestSelection cannot be combined with progressive exposure "
                "control, which selects items without the selection strategy"
            )

        self._current_theta = self.initial_theta
        self._current_se = float("inf")
        self._items_administered: list[int] = []
        self._responses: list[int] = []
        self._available_items: set[int] = set(range(model.n_items))
        self._theta_history: list[float] = []
        self._se_history: list[float] = []
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
        limit = item_limit(self._stopping, MaxItemsStop, CombinedStop)
        return n_items if limit is None else min(limit, n_items)

    def reset(self) -> None:
        """Reset the engine for a new examinee.

        Resetting an unused session reuses its exposure count. Once item
        selection begins, a reset opens and counts a new exposure session.
        """
        self._current_theta = self.initial_theta
        self._current_se = float("inf")
        reset_session_state(
            self,
            n_items=self.model.n_items,
            history_attrs=("_theta_history", "_se_history", "_info_history"),
        )

    def get_current_state(self) -> CATState:
        """Get the current state of the CAT session.

        Returns
        -------
        CATState
            Current state including theta, SE, items administered.
        """
        next_item = None
        if not self._is_complete and self._available_items:
            next_item = get_pending_item(self)

        return self._build_state(next_item)

    def _build_state(self, next_item: int | None) -> CATState:
        """Create a state snapshot with an explicitly resolved next item."""
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

        return get_pending_item(self)

    def _select_next_item(self) -> int:
        """Internal item selection with constraint handling."""
        strategy = self._selection
        # a-stratified strata and shadow tests span the engine's test horizon
        # unless the strategy was configured with its own test length.
        options: dict[str, Any] = (
            {"test_length": self._selection_horizon}
            if isinstance(strategy, (AStratified, ShadowTestSelection))
            and _accepts_test_length(strategy)
            else {}
        )
        return select_constrained_item(
            self,
            criteria=lambda eligible: strategy.get_item_criteria(
                self.model,
                self._current_theta,
                eligible,
                administered_items=self._items_administered,
                responses=self._responses,
                **options,
            ),
            select=lambda eligible: strategy.select_item(
                self.model,
                self._current_theta,
                eligible,
                self._items_administered,
                self._responses,
                **options,
            ),
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
            If the response is not an integer-valued numeric scalar in the
            selected item's category range. The selected item remains pending
            so a corrected response can be submitted.
        """
        if self._is_complete:
            raise RuntimeError("CAT session is already complete")

        item_idx, response = consume_pending_item(self, response)

        theta_arr = np.array([[self._current_theta]])
        record_item_administration(
            self,
            item_idx=item_idx,
            response=response,
            theta_arr=theta_arr,
        )

        self._update_theta()

        self._theta_history.append(self._current_theta)
        self._se_history.append(self._current_se)

        finalize_administered_item(self, self._build_state(next_item=None))

        return self.get_current_state()

    def _scoring_prior(
        self,
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        """Return the scoring prior's mean and covariance for the scorers.

        Each part is None when it takes its standard-normal default.
        """
        mean = None if self.prior_mean == 0.0 else np.array([float(self.prior_mean)])
        variance = float(self.prior_sd) ** 2
        cov = None if variance == 1.0 else np.array([[variance]])
        return mean, cov

    def _update_theta(self) -> None:
        """Update ability estimate based on administered items.

        A numerical scoring failure, such as a posterior without finite mass,
        warns and falls back to a logit of the proportion of the maximum
        score. Configuration errors propagate.
        """
        try:
            if self.scoring_method == "EAP":
                from mirt.cat._eap import score_binary_eap

                moments = score_binary_eap(self)
                if moments is not None:
                    theta, covariance = moments
                    self._current_theta = float(theta[0])
                    self._current_se = float(np.sqrt(covariance[0, 0]))
                    return
            result = score_administered_responses(
                self,
                bounds=self.theta_bounds,
            )
            self._current_theta = float(result.theta.ravel()[0])
            self._current_se = float(result.standard_error.ravel()[0])
        except (ArithmeticError, np.linalg.LinAlgError, ValueError) as error:
            if not is_numerical_scoring_failure(error):
                raise
            warn_scoring_failure(
                self, error, "using a logit of the proportion of the maximum score"
            )
            self._current_theta = self._simple_theta_estimate()
            self._current_se = self._simple_se_estimate()

    def _simple_theta_estimate(self) -> float:
        """Crude theta estimate: a logit of the proportion of the maximum score.

        A response in category ``k`` of an item with ``m`` categories scores
        ``k / (m - 1)``, so dichotomous items count correct answers.
        """
        if not self._responses:
            return self.initial_theta

        if self.model.is_polytomous:
            categories = np.asarray(self.model.n_categories)[self._items_administered]
            p = float(np.sum(self._responses) / np.sum(categories - 1))
        else:
            p = float(np.mean(self._responses))
        p = np.clip(p, PROB_CLIP_MIN, PROB_CLIP_MAX)

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
            polytomous=bool(self.model.is_polytomous),
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
        theta = validate_simulation_values(true_theta, name="true_theta")
        if theta.ndim != 0:
            raise ValueError("true_theta must be a finite scalar")

        return run_simulation_loop(
            self,
            float(theta),
            response_generator=response_generator,
        )

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
            p = float(prob.ravel()[0])
            return int(self.rng.random() < p)
        else:
            probs = prob.ravel()
            return int(self.rng.choice(len(probs), p=probs))

    def run_batch_simulation(
        self,
        true_thetas: NDArray[np.float64] | list[float],
        n_replications: int = 1,
        use_rust: bool = True,
        vectorized: bool = True,
    ) -> list[CATResult]:
        """Simulate CAT for multiple examinees.

        Three implementations share one result contract, tried in order:

        1. The Rust backend, for matching 1PL/2PL, EAP, MFI, and SE-stop
           configurations when ``use_rust`` is True. Its results omit the
           estimate and information histories.
        2. Lock-step simulation when ``vectorized`` is True, for unmodified
           engines using MFI selection, EAP scoring, SE and/or maximum-length
           stopping, no exposure or content control, and a built-in
           unidimensional 1PL-4PL, GRM, GPCM, or PCM model. All examinees
           advance together with one vectorized step per item position.
        3. Independent sessions through :meth:`run_simulation`, the reference
           implementation used for every other configuration.

        Given the same responses, lock step reproduces the reference item
        paths, estimates, standard errors, stopping reasons, and histories.
        Both paths turn one uniform draw from the engine's generator into
        each response, but the reference starts each examinee's draws right
        after the previous examinee's last one, while lock step reserves
        ``L`` draws per examinee, where ``L`` is ``max_items`` capped at the
        pool size. A seeded engine therefore gives identical results on both
        paths when every examinee answers ``L`` items (fixed-length tests)
        and other, equally distributed, responses when tests end early.
        The Rust backend draws its responses from its own generator, seeded
        for each call from the engine's generator. On every path a seeded
        engine is reproducible, and successive calls on one engine continue
        its random stream instead of repeating earlier sessions.
        The Rust and lock-step paths leave the engine's interactive session
        state unchanged.

        Parameters
        ----------
        true_thetas : array-like
            Array of true ability values.
        n_replications : int, optional
            Number of replications per theta value. Default is 1.
        use_rust : bool, optional
            Use the Rust backend when the configuration supports it.
            Default is True.
        vectorized : bool, optional
            Use lock-step simulation when the configuration supports it.
            Set False to run independent sessions with the engine's random
            stream. Default is True.

        Returns
        -------
        list[CATResult]
            Results for all simulations, with all replications of the first
            theta first.
        """
        thetas, n_replications, use_rust, vectorized = _validate_batch_controls(
            true_thetas,
            n_replications,
            use_rust,
            vectorized,
        )

        if should_use_rust(use_rust) and self._can_use_rust_simulation():
            native_results = self._run_batch_rust(thetas, n_replications)
            if native_results is not None:
                return native_results

        if vectorized and supports_lockstep(self):
            lockstep_results = simulate_lockstep(self, thetas, n_replications)
            if lockstep_results is not None:
                return lockstep_results

        return [
            self.run_simulation(float(theta))
            for theta in thetas
            for _ in range(n_replications)
        ]

    def _can_use_rust_simulation(self) -> bool:
        """Check if Rust simulation can be used."""
        from mirt.models import OneParameterLogistic, TwoParameterLogistic

        if not uses_native_defaults(self, CATEngine) or not uses_native_defaults(
            self._selection, MaxFisherInformation
        ):
            return False
        if self.scoring_method != "EAP" or self.initial_theta != 0.0:
            return False
        if (
            isinstance(self.n_quadpts, (bool, np.bool_))
            or not isinstance(self.n_quadpts, (int, np.integer))
            or self.n_quadpts < 5
        ):
            return False
        if not uses_native_defaults(
            self._exposure, NoExposureControl
        ) or not uses_native_defaults(self._content, NoContentConstraint):
            return False
        if type(self.model) not in (OneParameterLogistic, TwoParameterLogistic):
            return False
        if not uses_builtin_model_hooks(
            self.model, likelihood=True
        ) or not uses_original_model_hook(self.model, "information"):
            return False

        params = self.model.parameters
        if "discrimination" not in params or "difficulty" not in params:
            return False

        if self.model.n_factors != 1:
            return False

        disc = params["discrimination"]
        if disc.ndim != 1:
            return False

        return self._native_stopping_parameters() is not None

    def _get_quadrature(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the scoring prior's quadrature points and weights."""
        from mirt.scoring._common import build_quadrature

        prior_mean, prior_cov = self._scoring_prior()
        points, weights = build_quadrature(
            n_quadpts=self.n_quadpts,
            n_factors=1,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )
        return points.ravel(), weights.ravel()

    def _native_stopping_parameters(
        self, *, require_standard_error: bool = True
    ) -> tuple[float, int, int] | None:
        """Translate supported stopping rules into native controls.

        Without ``require_standard_error``, maximum-length rules alone are
        accepted too and the returned SE threshold is ``-inf``.
        """
        if type(self._stopping) is CombinedStop:
            if not uses_native_defaults(self._stopping, CombinedStop):
                return None
            if self._stopping.operator != "or":
                return None
            rules = self._stopping.rules
            min_items = self._stopping.min_items
        else:
            rules = [self._stopping]
            min_items = 1

        if isinstance(min_items, (bool, np.bool_)) or not isinstance(
            min_items, Integral
        ):
            return None
        se_threshold: float | None = None
        max_items = self.model.n_items
        for rule in rules:
            if uses_native_defaults(rule, StandardErrorStop):
                if se_threshold is not None:
                    return None
                if isinstance(rule.threshold, (bool, np.bool_)) or not isinstance(
                    rule.threshold, Real
                ):
                    return None
                try:
                    se_threshold = float(rule.threshold)
                except (ValueError, OverflowError):
                    return None
                if not np.isfinite(se_threshold) or se_threshold <= 0.0:
                    return None
            elif uses_native_defaults(rule, MaxItemsStop):
                if (
                    isinstance(rule.max_items, (bool, np.bool_))
                    or not isinstance(rule.max_items, Integral)
                    or rule.max_items < 1
                ):
                    return None
                max_items = min(max_items, int(rule.max_items))
            else:
                return None

        if se_threshold is None:
            if require_standard_error:
                return None
            se_threshold = -np.inf
        if min_items < 1 or min_items > max_items:
            return None
        return se_threshold, max_items, min_items

    def _get_stopping_parameters(self) -> tuple[float, int, int]:
        """Return native stopping controls after eligibility is established."""
        parameters = self._native_stopping_parameters()
        if parameters is None:
            raise RuntimeError("stopping rules are not supported by native simulation")
        return parameters

    def _native_calls(
        self,
        true_thetas: NDArray[np.float64],
        n_replications: int,
        kernel: Any,
        chunk: int,
    ) -> Iterator[tuple[int, int, int, Any]] | None:
        """Return native results for every session, in theta-major order.

        The kernels give replication ``r`` of theta ``t`` its own random
        stream, mixed from the call's seed, ``t`` and ``r``. Each call takes
        its seed from the engine's generator, so neighbouring engine seeds do
        not replay each other's sessions and successive calls advance the
        generator as the Python paths do. A call covers at most ``chunk``
        sessions: a block of whole thetas, or a block of one theta's
        replications when they alone exceed ``chunk``.
        ``kernel(thetas, n_replications, seed)`` runs one call; each item
        yields the index of the call's first theta, its number of thetas, its
        replications per theta and its payload. Returns None, with the
        generator restored, when the backend is unavailable.
        """
        n_thetas = len(true_thetas)
        if n_replications <= chunk:
            per_call = chunk // n_replications
            blocks = [
                (start, min(per_call, n_thetas - start), n_replications)
                for start in range(0, n_thetas, per_call)
            ]
        else:
            blocks = [
                (theta, 1, min(chunk, n_replications - done))
                for theta in range(n_thetas)
                for done in range(0, n_replications, chunk)
            ]

        def call(block: tuple[int, int, int]) -> Any:
            start, count, replications = block
            seed = int(self.rng.integers(0, 2**64, dtype=np.uint64))
            return kernel(true_thetas[start : start + count], replications, seed)

        state = self.rng.bit_generator.state
        first = call(blocks[0])
        if first is None:
            self.rng.bit_generator.state = state
            return None

        def calls() -> Iterator[tuple[int, int, int, Any]]:
            yield (*blocks[0], first)
            for block in blocks[1:]:
                payload = call(block)
                if payload is None:
                    raise RuntimeError("native CAT simulation became unavailable")
                yield (*block, payload)

        return calls()

    def _native_controls(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        int,
        int,
    ]:
        """Return the item bank, quadrature, and stopping controls."""
        params = self.model.parameters
        disc = params["discrimination"].astype(np.float64)
        diff = params["difficulty"].astype(np.float64)
        quad_points, quad_weights = self._get_quadrature()
        return (disc, diff, quad_points, quad_weights, *self._get_stopping_parameters())

    def _run_batch_rust(
        self,
        true_thetas: NDArray[np.float64],
        n_replications: int,
    ) -> list[CATResult] | None:
        """Run batch simulation using Rust backend, or None if it is unavailable."""
        controls = self._native_controls()
        width = max(1, min(int(controls[5]), self.model.n_items))
        calls = self._native_calls(
            true_thetas,
            n_replications,
            lambda thetas, replications, seed: rust_cat_simulate_batch_full(
                thetas, *controls, replications, seed
            ),
            max(1, _NATIVE_PATH_VALUES // width),
        )
        if calls is None:
            return None

        # Preserve authored rule priority and distinguish an exhausted pool
        # from a configured maximum. Evaluating an isolated copy also leaves
        # the engine's previous interactive stopping state intact.
        stopping = deepcopy(self._stopping)
        results = []
        for *_, payload in calls:
            theta_est, se_est, n_items, _, item_paths, response_paths = payload
            for i in range(len(theta_est)):
                count = int(n_items[i])
                items = np.asarray(item_paths[i, :count], dtype=np.int_).tolist()
                responses = np.asarray(response_paths[i, :count], dtype=np.int_)
                stopping.reset()
                state = CATState(
                    theta=float(theta_est[i]),
                    standard_error=float(se_est[i]),
                    items_administered=items,
                    responses=responses.tolist(),
                    n_items=count,
                )
                stopping_reason = (
                    stopping.get_reason()
                    if stopping.should_stop(state)
                    else "Item pool exhausted"
                )
                results.append(
                    CATResult(
                        theta=float(theta_est[i]),
                        standard_error=float(se_est[i]),
                        items_administered=items,
                        responses=responses,
                        n_items_administered=count,
                        stopping_reason=stopping_reason,
                        theta_history=[],
                        se_history=[],
                        item_info_history=[],
                        polytomous=False,
                    )
                )

        return results

    def _native_error_moments(
        self,
        true_thetas: NDArray[np.float64],
        n_replications: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
        """Return native conditional bias, MSE, and mean test length.

        Each call reports every theta's mean error, squared error and length
        over its replications, which are pooled across calls.
        """
        controls = self._native_controls()
        calls = self._native_calls(
            true_thetas,
            n_replications,
            lambda thetas, replications, seed: rust_cat_conditional_mse(
                thetas, *controls, replications, seed
            ),
            _NATIVE_MSE_SESSIONS,
        )
        if calls is None:
            return None
        sums = np.zeros((3, len(true_thetas)))
        for start, count, replications, (_, *moments) in calls:
            sums[:, start : start + count] += replications * np.asarray(
                moments, dtype=np.float64
            )
        bias, mse, avg_items = sums / n_replications
        return bias, mse, avg_items

    def compute_conditional_mse(
        self,
        true_thetas: NDArray[np.float64] | list[float],
        n_replications: int = 100,
        use_rust: bool = True,
        vectorized: bool = True,
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
            Available for matching 1PL/2PL, EAP, MFI, and SE-stop configurations.
        vectorized : bool, optional
            Use lock-step simulation when the configuration supports it, as
            described in :meth:`run_batch_simulation`. Default is True.

        Notes
        -----
        On the lock-step path, a seeded engine simulates the same sessions as
        ``run_batch_simulation`` with the same arguments on an engine with the
        same seed. The Rust path seeds every session independently from the
        engine's generator, as ``run_batch_simulation`` does.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, NDArray]
            Tuple of (thetas, biases, MSEs, avg_items).
        """
        thetas, n_replications, use_rust, vectorized = _validate_batch_controls(
            true_thetas,
            n_replications,
            use_rust,
            vectorized,
        )

        if should_use_rust(use_rust) and self._can_use_rust_simulation():
            native = self._native_error_moments(thetas, n_replications)
            if native is not None:
                return (thetas, *native)

        if vectorized and supports_lockstep(self):
            moments = lockstep_error_moments(self, thetas, n_replications)
            if moments is not None:
                return (thetas, *moments)

        biases = np.zeros(len(thetas))
        mses = np.zeros(len(thetas))
        avg_items = np.zeros(len(thetas))

        for i, true_theta in enumerate(thetas):
            bias, mse, mean_items = simulate_error_moments(
                self, float(true_theta), n_replications=n_replications
            )
            biases[i] = bias
            mses[i] = mse
            avg_items[i] = mean_items

        return thetas, biases, mses, avg_items

    def __repr__(self) -> str:
        status = "complete" if self._is_complete else "in_progress"
        return (
            f"CATEngine(n_items={self.model.n_items}, "
            f"administered={len(self._items_administered)}, "
            f"status={status})"
        )
