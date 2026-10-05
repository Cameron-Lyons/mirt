"""MCMC and MHRM Estimation for IRT Models.

This module provides stochastic estimation methods:
- MHRM (Metropolis-Hastings Robbins-Monro)
- Gibbs Sampling for full Bayesian inference

Uses fast Rust backend when available for 2PL models.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtEstimationError, MirtValidationError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

from mirt.estimation.base import BaseEstimator, _reject_parameter_restrictions
from mirt.results.fit_result import FitResult

PosteriorValue = NDArray[np.float64] | np.float64
PosteriorSummary = dict[str, dict[str, PosteriorValue]]
CredibleIntervals = dict[str, tuple[PosteriorValue, PosteriorValue]]

_Objective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]

_MHRM_GAIN_SEQUENCES = ("standard", "adaptive")
# Ability sweeps before the first parameter step.
_MHRM_WARMUP_SWEEPS = 30
# Largest change of any item coordinate in one cycle; narrower boxes allow
# a quarter of their width.
_MHRM_MAX_STEP = 1.0
_MHRM_MAX_HALVINGS = 10
# Relative step for differentiating item gradients.
_MHRM_DIFFERENCE_STEP = 1e-5
_MHRM_RELATIVE_CURVATURE_FLOOR = 1e-8


def _validate_count(value: int, name: str, minimum: int) -> int:
    """Return an integer sampler control of at least ``minimum``."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        expected = "positive integer" if minimum == 1 else "non-negative integer"
        raise MirtValidationError(
            f"{name} must be a {expected}",
            parameter=name,
            value=value,
            expected=expected,
        )
    return int(value)


def _is_2pl_unidimensional(model: BaseItemModel) -> bool:
    """Check if model is 2PL unidimensional."""
    return (
        model.model_name == "2PL"
        and hasattr(model, "n_factors")
        and model.n_factors == 1
    )


@dataclass
class MCMCResult:
    """Result from MCMC estimation.

    Attributes
    ----------
    model : BaseItemModel
        Fitted model with posterior mean parameters
    chains : dict
        MCMC chains for each parameter
    log_likelihood : float
        Log-likelihood at posterior mean
    dic : float
        Deviance Information Criterion
    waic : float
        Watanabe-Akaike Information Criterion
    rhat : dict
        Gelman-Rubin convergence diagnostics
    ess : dict
        Effective sample sizes
    """

    model: Any
    chains: dict[str, NDArray[np.float64]]
    log_likelihood: float
    dic: float
    waic: float
    rhat: dict[str, float]
    ess: dict[str, float]
    n_iterations: int
    burnin: int
    thin: int

    @staticmethod
    def _validate_credible_level(credible_level: float) -> float:
        """Return a finite credible level strictly between zero and one."""
        if (
            isinstance(credible_level, bool)
            or not isinstance(credible_level, Real)
            or not np.isfinite(credible_level)
            or not 0.0 < credible_level < 1.0
        ):
            raise ValueError("credible_level must be between 0 and 1")
        return float(credible_level)

    def _selected_chains(
        self,
        parameters: str | Sequence[str] | None,
    ) -> dict[str, NDArray[np.float64]]:
        """Select and validate posterior chains for result summaries."""
        if parameters is None:
            names = tuple(self.chains)
        elif isinstance(parameters, str):
            names = (parameters,)
        else:
            try:
                names = tuple(parameters)
            except TypeError as exc:
                raise ValueError(
                    "parameters must be a chain name or sequence of chain names"
                ) from exc
            if not all(isinstance(name, str) for name in names):
                raise ValueError("parameters must contain only chain names")

        unknown = tuple(
            dict.fromkeys(name for name in names if name not in self.chains)
        )
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(f"unknown posterior chain: {joined}")

        selected: dict[str, NDArray[np.float64]] = {}
        n_draws: int | None = None
        for name in dict.fromkeys(names):
            try:
                chain = np.asarray(self.chains[name], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"posterior chain '{name}' must contain numeric values"
                ) from exc
            if chain.ndim == 0 or chain.size == 0 or chain.shape[0] == 0:
                raise ValueError(
                    f"posterior chain '{name}' must contain at least one draw"
                )
            if not np.all(np.isfinite(chain)):
                raise ValueError(
                    f"posterior chain '{name}' must contain only finite values"
                )
            if n_draws is None:
                n_draws = chain.shape[0]
            elif chain.shape[0] != n_draws:
                raise ValueError(
                    "selected posterior chains must have equal draw counts"
                )
            selected[name] = chain

        return selected

    def posterior_summary(
        self,
        credible_level: float = 0.95,
        parameters: str | Sequence[str] | None = None,
    ) -> PosteriorSummary:
        """Return posterior moments and equal-tailed credible intervals.

        Statistics are computed over the leading draw dimension. Remaining
        dimensions are preserved, so item parameters and person abilities can
        be summarized with the same method.

        Parameters
        ----------
        credible_level : float
            Probability covered by each equal-tailed interval.
        parameters : str or sequence of str, optional
            Chain names to summarize. By default, all stored chains are used.

        Returns
        -------
        dict
            Mean, standard deviation, median, and interval bounds for each
            selected chain.
        """
        level = self._validate_credible_level(credible_level)
        selected = self._selected_chains(parameters)
        tail = (1.0 - level) / 2.0

        result: PosteriorSummary = {}
        for name, chain in selected.items():
            lower, median, upper = np.quantile(
                chain,
                (tail, 0.5, 1.0 - tail),
                axis=0,
            )
            result[name] = {
                "mean": np.mean(chain, axis=0),
                "std": np.std(chain, axis=0),
                "median": median,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        return result

    def credible_intervals(
        self,
        credible_level: float = 0.95,
        parameters: str | Sequence[str] | None = None,
    ) -> CredibleIntervals:
        """Return equal-tailed credible intervals for selected chains."""
        level = self._validate_credible_level(credible_level)
        selected = self._selected_chains(parameters)
        tail = (1.0 - level) / 2.0
        intervals: CredibleIntervals = {}
        for name, chain in selected.items():
            lower, upper = np.quantile(chain, (tail, 1.0 - tail), axis=0)
            intervals[name] = (lower, upper)
        return intervals

    def summary(self) -> str:
        """Generate summary of MCMC results."""
        lines = [
            "MCMC Estimation Summary",
            "=" * 50,
            f"Iterations: {self.n_iterations}",
            f"Burnin: {self.burnin}",
            f"Thinning: {self.thin}",
            "",
            f"Log-likelihood: {self.log_likelihood:.4f}",
            f"DIC: {self.dic:.4f}",
            f"WAIC: {self.waic:.4f}",
            "",
            "Convergence (R-hat):",
        ]

        for name, rhat in self.rhat.items():
            status = "OK" if rhat < 1.1 else "WARNING"
            lines.append(f"  {name}: {rhat:.4f} ({status})")

        if self.ess:
            lines.extend(("", "Effective sample size:"))
            for name, ess in self.ess.items():
                lines.append(f"  {name}: {ess:.1f}")

        return "\n".join(lines)


class MHRMEstimator(BaseEstimator):
    """Metropolis-Hastings Robbins-Monro estimator (Cai, 2010).

    Each cycle draws abilities with a random-walk Metropolis step and then
    moves every item along its complete-data score ``s_k``, preconditioned by
    a stochastic approximation of the complete-data information:

    .. math::

        \\Gamma_k = \\Gamma_{k-1} + g_k (H_k - \\Gamma_{k-1}), \\qquad
        \\beta_{k+1} = \\beta_k + g_k \\Gamma_k^{-1} s_k,

    where ``H_k`` is the item's complete-data information at the current
    draw. Burn-in cycles use ``g_k = 1``, Newton steps on the imputed data;
    later gains decrease (see ``gain_sequence``) and the estimates average the
    post-burn-in iterates. Items are updated blockwise, so any built-in item
    family, including polytomous and multidimensional ones, is supported.

    Uses fast parallel Rust backend for 2PL models when available.

    References
    ----------
    Cai, L. (2010). Metropolis-Hastings Robbins-Monro algorithm for
    confirmatory item factor analysis. Journal of Educational and
    Behavioral Statistics, 35(3), 307-335.
    """

    def __init__(
        self,
        n_cycles: int = 2000,
        burnin: int = 500,
        n_chains: int = 1,
        proposal_sd: float = 0.5,
        gain_sequence: str = "standard",
        verbose: bool = False,
        use_rust: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize MHRM estimator.

        Parameters
        ----------
        n_cycles : int
            Number of MHRM cycles
        burnin : int
            Number of initial unit-gain cycles, which are excluded from the
            parameter average; the final iterate is used when
            ``burnin >= n_cycles``
        n_chains : int
            Number of parallel chains
        proposal_sd : float
            Standard deviation for MH proposals
        gain_sequence : str
            Post-burn-in gains for the ``t``-th cycle after burn-in:
            ``1 / (t + 1)`` ('standard') or ``min(1, 10 / (t + 10))``
            ('adaptive')
        verbose : bool
            Whether to print progress
        use_rust : bool
            Whether to use Rust backend when available
        seed : int, optional
            Random seed for reproducibility
        """
        n_cycles = _validate_count(n_cycles, "n_cycles", 1)
        burnin = _validate_count(burnin, "burnin", 0)
        if (
            isinstance(proposal_sd, (bool, np.bool_))
            or not isinstance(proposal_sd, (int, float, np.integer, np.floating))
            or not np.isfinite(proposal_sd)
            or proposal_sd <= 0
        ):
            raise MirtValidationError(
                "proposal_sd must be finite and positive",
                parameter="proposal_sd",
                value=proposal_sd,
                expected="> 0",
            )
        if gain_sequence not in _MHRM_GAIN_SEQUENCES:
            raise MirtValidationError(
                "gain_sequence must be 'standard' or 'adaptive'",
                parameter="gain_sequence",
                value=gain_sequence,
                expected="'standard' or 'adaptive'",
            )
        super().__init__(max_iter=n_cycles, tol=1e-4, verbose=verbose)
        self.n_cycles = n_cycles
        self.burnin = burnin
        self.n_chains = n_chains
        self.proposal_sd = float(proposal_sd)
        self.gain_sequence = gain_sequence
        self.use_rust = use_rust
        self.seed = seed

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        **kwargs: Any,
    ) -> FitResult:
        """Fit model using MHRM algorithm.

        Parameters
        ----------
        model : BaseItemModel
            IRT model to fit
        responses : NDArray
            Response matrix (n_persons, n_items)
        **kwargs
            Additional arguments (prior_mean, prior_cov)

        Returns
        -------
        FitResult
            Fitted model result

        Raises
        ------
        MirtValidationError
            If ``set_free_parameter_masks`` fixes parameters, which MHRM
            cannot hold.
        """
        from mirt._backend_config import should_use_rust
        from mirt.backends.rust.estimation import mhrm_fit_2pl

        _reject_parameter_restrictions(model, "MHRMEstimator")
        responses = self._validate_responses(responses, model.n_items)
        n_persons, n_items = responses.shape

        if should_use_rust(self.use_rust) and _is_2pl_unidimensional(model):
            seed = (
                self.seed
                if self.seed is not None
                else np.random.default_rng().integers(0, 2**31)
            )

            discrimination, difficulty, _ = mhrm_fit_2pl(
                responses,
                n_cycles=self.n_cycles,
                burnin=self.burnin,
                proposal_sd=self.proposal_sd,
                seed=seed,
                gain_sequence=self.gain_sequence,
            )

            if not model._parameters:
                model._initialize_parameters()
            model._parameters["discrimination"] = np.asarray(discrimination)
            model._parameters["difficulty"] = np.asarray(difficulty)
            model._is_fitted = True

            # Score at MAP abilities as the NumPy path does, so AIC and BIC do
            # not depend on the backend.
            theta_map = self._estimate_theta_map(
                model, responses, np.random.default_rng(seed)
            )
            log_likelihood = float(np.sum(model.log_likelihood(responses, theta_map)))

            n_params = 2 * n_items
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + np.log(n_persons) * n_params

            from mirt.backends.rust.diagnostics import compute_item_se_parallel
            from mirt.backends.rust.estep import e_step_complete
            from mirt.estimation.quadrature import GaussHermiteQuadrature

            disc = np.asarray(discrimination)
            diff = np.asarray(difficulty)
            quad = GaussHermiteQuadrature(n_points=21, n_dimensions=1)
            posterior_weights, _ = e_step_complete(
                responses,
                quad.nodes.ravel(),
                quad.weights.ravel(),
                disc,
                diff,
            )
            se_a, se_b = compute_item_se_parallel(
                responses,
                posterior_weights,
                quad.nodes.ravel(),
                disc,
                diff,
            )

            return FitResult(
                model=model,
                log_likelihood=log_likelihood,
                n_iterations=self.n_cycles,
                converged=True,
                standard_errors={
                    "discrimination": np.asarray(se_a),
                    "difficulty": np.asarray(se_b),
                },
                aic=aic,
                bic=bic,
                n_observations=n_persons * n_items,
                n_parameters=n_params,
            )

        if not model._parameters:
            model._initialize_parameters()

        rng = np.random.default_rng(self.seed)
        theta = np.zeros((n_persons, model.n_factors))
        # Let the ability chain leave its degenerate start before the first
        # unit-gain step, whose complete-data information needs spread draws.
        for _ in range(_MHRM_WARMUP_SWEEPS):
            theta = self._sample_theta(model, responses, theta, rng)

        information: list[NDArray[np.float64] | None] = [None] * n_items
        param_history: dict[str, list] = {name: [] for name in model.parameters}

        for cycle in range(self.n_cycles):
            theta = self._sample_theta(model, responses, theta, rng)

            gain = self._compute_gain(cycle)
            self._update_parameters(model, responses, theta, gain, information)

            if cycle >= self.burnin:
                for name, values in model.parameters.items():
                    param_history[name].append(values.copy())

            if self.verbose and (cycle + 1) % 100 == 0:
                ll = np.sum(model.log_likelihood(responses, theta))
                print(f"Cycle {cycle + 1}/{self.n_cycles}: LL = {ll:.4f}")

        for name in model.parameters:
            if param_history[name]:
                model._parameters[name] = np.mean(param_history[name], axis=0)

        model._is_fitted = True

        theta_final = self._estimate_theta_map(model, responses, rng)
        ll = float(np.sum(model.log_likelihood(responses, theta_final)))

        se = {}
        for name, chain in param_history.items():
            if chain:
                se[name] = np.std(chain, axis=0)

        return FitResult(
            model=model,
            log_likelihood=ll,
            n_iterations=self.n_cycles,
            converged=True,
            standard_errors=se,
            aic=-2 * ll + 2 * self._count_parameters(model),
            bic=-2 * ll + np.log(n_persons) * self._count_parameters(model),
            n_observations=n_persons * n_items,
            n_parameters=self._count_parameters(model),
        )

    def _sample_theta(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Metropolis-Hastings step for theta sampling."""
        n_persons = theta.shape[0]

        proposal = theta + rng.normal(0, self.proposal_sd, theta.shape)

        ll_current = model.log_likelihood(responses, theta)
        ll_proposal = model.log_likelihood(responses, proposal)

        prior_current = stats.norm.logpdf(theta).sum(axis=1)
        prior_proposal = stats.norm.logpdf(proposal).sum(axis=1)

        log_alpha = (ll_proposal + prior_proposal) - (ll_current + prior_current)
        log_u = np.log(rng.random(n_persons))

        accept = log_u < log_alpha
        theta_new = np.where(accept[:, None], proposal, theta)

        return theta_new

    def _compute_gain(self, cycle: int) -> float:
        """Return the Robbins-Monro gain: one in burn-in, then decreasing."""
        if cycle < self.burnin:
            return 1.0
        t = cycle - self.burnin
        if self.gain_sequence == "adaptive":
            return min(1.0, 10.0 / (t + 10))
        return 1.0 / (t + 1)

    def _update_parameters(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        gain: float,
        information: list[NDArray[np.float64] | None],
    ) -> None:
        """Take one preconditioned Robbins-Monro step on every item.

        ``information`` holds each item's running complete-data information
        and is updated in place. Fixed coordinates never enter the step. A
        step that would lower the item's complete-data likelihood on the
        current draw is halved until it does not, which keeps unit-gain
        burn-in steps from overshooting into degenerate regions; graded
        thresholds must also stay ordered.
        """
        from mirt.estimation.em import _graded_threshold_constraint

        updates: dict[int, NDArray[np.float64]] = {}
        for item in range(model.n_items):
            observed = responses[:, item] >= 0
            if not observed.any():
                continue
            params, bounds = self._get_item_params_and_bounds(model, item)
            if params.size == 0:
                continue
            objective = self._item_objective(
                model, item, theta[observed], responses[observed, item], params, bounds
            )
            loss, gradient = objective(params)
            hessian = _gradient_jacobian(objective, params, gradient, bounds)
            previous = information[item]
            current = (
                hessian if previous is None else previous + gain * (hessian - previous)
            )
            information[item] = current
            lower, upper = np.asarray(bounds, dtype=np.float64).T
            step = _robbins_monro_step(current, gradient, params, lower, upper, gain)
            if not np.all(np.isfinite(step)):
                continue
            ordering = _graded_threshold_constraint(model, item, params.size)
            for _ in range(_MHRM_MAX_HALVINGS):
                trial = np.clip(params + step, lower, upper)
                if (
                    ordering is None or np.all(ordering.A @ trial >= ordering.lb)
                ) and objective(trial)[0] <= loss:
                    updates[item] = trial
                    break
                step *= 0.5
        if updates:
            _set_item_vectors(model, updates)

    def _item_objective(
        self,
        model: BaseItemModel,
        item: int,
        theta: NDArray[np.float64],
        responses: NDArray[np.int_],
        params: NDArray[np.float64],
        bounds: list[tuple[float, float]],
    ) -> _Objective:
        """Return an item's complete-data loss and gradient on the draws.

        Built-in families use their analytic kernels. Custom models evaluate
        their public probability hook and differentiate it numerically.
        """
        from mirt.estimation._mc_objective import _item_kernel

        # Kernels may skip overflow guards inside their box, so the box must
        # cover the current values and the differencing steps around them.
        margin = 4 * _MHRM_DIFFERENCE_STEP * np.maximum(1.0, np.abs(params))
        box = [
            (min(low, value - pad), max(high, value + pad))
            for (low, high), value, pad in zip(bounds, params, margin, strict=True)
        ]
        kernel = _item_kernel(
            model, item, theta, responses, np.ones(len(responses)), box
        )
        if kernel is not None:
            return kernel

        rows = np.arange(len(responses))
        lower, upper = np.asarray(bounds, dtype=np.float64).T

        def loss(trial: NDArray[np.float64]) -> float:
            self._set_item_params(model, item, trial)
            probabilities = np.asarray(model.probability(theta, item), np.float64)
            if not model.is_polytomous:
                probabilities = probabilities.reshape(-1)
                probabilities = np.column_stack((1.0 - probabilities, probabilities))
            chosen = probabilities[rows, responses]
            if not np.all(np.isfinite(chosen)):
                raise MirtEstimationError("model returned invalid item probabilities")
            return -float(np.sum(np.log(np.clip(chosen, PROB_EPSILON, 1.0))))

        def objective(
            trial: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            original, _ = self._get_item_params_and_bounds(model, item)
            try:
                value = loss(trial)
                gradient = np.empty_like(trial)
                for k in range(trial.size):
                    step = _MHRM_DIFFERENCE_STEP * max(1.0, abs(trial[k]))
                    forward, backward = trial.copy(), trial.copy()
                    forward[k] = min(trial[k] + step, max(upper[k], trial[k]))
                    backward[k] = max(trial[k] - step, min(lower[k], trial[k]))
                    gradient[k] = (loss(forward) - loss(backward)) / (
                        forward[k] - backward[k]
                    )
            finally:
                self._set_item_params(model, item, original)
            return value, gradient

        return objective

    def _estimate_theta_map(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Estimate theta using MAP with current parameters."""
        n_persons = responses.shape[0]
        theta = rng.standard_normal((n_persons, model.n_factors))

        for _ in range(50):
            ll = model.log_likelihood(responses, theta)
            prior = -0.5 * np.sum(theta**2, axis=1)

            h = 1e-4
            grad = np.zeros_like(theta)
            for d in range(model.n_factors):
                theta_plus = theta.copy()
                theta_plus[:, d] += h
                ll_plus = model.log_likelihood(responses, theta_plus)
                prior_plus = -0.5 * np.sum(theta_plus**2, axis=1)
                grad[:, d] = (ll_plus + prior_plus - ll - prior) / h

            theta = theta + 0.1 * grad

        return theta

    def _count_parameters(self, model: BaseItemModel) -> int:
        """Count number of parameters."""
        return model.n_parameters


def _set_item_vectors(
    model: BaseItemModel,
    updates: dict[int, NDArray[np.float64]],
) -> None:
    """Write several items' free coordinates with one validated update.

    Vectors use the layout of ``BaseEstimator._get_item_params_and_bounds``.
    """
    values = model.parameters
    masks = model.free_parameter_masks
    offsets = dict.fromkeys(updates, 0)
    changed: dict[str, NDArray[np.float64]] = {}
    for name, array in values.items():
        if array.ndim == 0 or array.shape[0] != model.n_items:
            continue
        rows = np.array(array, dtype=np.float64).reshape(model.n_items, -1)
        free = np.asarray(masks[name], dtype=np.bool_).reshape(model.n_items, -1)
        for item, vector in updates.items():
            n_free = int(np.count_nonzero(free[item]))
            rows[item, free[item]] = vector[offsets[item] : offsets[item] + n_free]
            offsets[item] += n_free
        canonical = model._canonical_parameter_values(name, rows.reshape(array.shape))
        if not np.array_equal(canonical, model._parameters[name]):
            changed[name] = canonical
    model.set_parameters(**changed)


def _gradient_jacobian(
    objective: _Objective,
    params: NDArray[np.float64],
    gradient: NDArray[np.float64],
    bounds: list[tuple[float, float]],
) -> NDArray[np.float64]:
    """Differentiate a loss gradient by one-sided differences, symmetrized.

    Each coordinate steps toward the inside of its box, so trial values never
    leave the region where the model is defined.
    """
    size = params.size
    jacobian = np.empty((size, size))
    for k in range(size):
        step = _MHRM_DIFFERENCE_STEP * max(1.0, abs(params[k]))
        if params[k] + step > bounds[k][1]:
            step = -step
        trial = params.copy()
        trial[k] += step
        jacobian[:, k] = (objective(trial)[1] - gradient) / step
    return 0.5 * (jacobian + jacobian.T)


def _robbins_monro_step(
    information: NDArray[np.float64],
    gradient: NDArray[np.float64],
    params: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    gain: float,
) -> NDArray[np.float64]:
    """Return the preconditioned item step for a loss gradient.

    Coordinates on a bound that the step would push outward are held, and the
    rest solve their own information block. The step is then shrunk so no
    coordinate moves farther than a quarter of its box, or one unit.
    """
    held = ((params <= lower) & (gradient > 0)) | ((params >= upper) & (gradient < 0))
    free = ~held
    step = np.zeros_like(params)
    step[free] = gain * _precondition(information[np.ix_(free, free)], -gradient[free])
    limits = np.minimum(_MHRM_MAX_STEP, 0.25 * (upper - lower))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = float(np.max(np.abs(step) / limits, initial=0.0))
    if ratio > 1.0:
        step /= ratio
    return step


def _precondition(
    information: NDArray[np.float64],
    score: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve ``information @ step = score`` with curvature kept positive.

    Eigenvalues are replaced by their magnitudes and floored relative to the
    largest one, so flat or indefinite directions take bounded descent steps.
    """
    values, vectors = np.linalg.eigh(information)
    scale = float(np.max(np.abs(values), initial=0.0))
    if not np.isfinite(scale):
        return np.full_like(score, np.nan)
    floor = max(_MHRM_RELATIVE_CURVATURE_FLOOR * scale, np.finfo(np.float64).tiny)
    return vectors @ ((vectors.T @ score) / np.maximum(np.abs(values), floor))


class GibbsSampler(BaseEstimator):
    """Full Bayesian estimation via Gibbs sampling.

    Implements blocked Gibbs sampling where:
    1. Sample theta | parameters, data
    2. Sample parameters | theta, data

    This provides full posterior distributions for all parameters.

    Uses fast parallel Rust backend for 2PL models when available.
    With ``n_chains > 1`` the draws of several seeded chains are stacked;
    ``parallel_chains=True`` runs NumPy chains in worker processes.
    """

    def __init__(
        self,
        n_iter: int = 5000,
        burnin: int = 1000,
        thin: int = 1,
        n_chains: int = 1,
        priors: dict[str, Any] | None = None,
        verbose: bool = False,
        use_rust: bool = True,
        seed: int | None = None,
        parallel_chains: bool = False,
    ) -> None:
        """Initialize Gibbs sampler.

        Parameters
        ----------
        n_iter : int
            Number of iterations
        burnin : int
            Burnin iterations; must be less than ``n_iter``
        thin : int
            Thinning interval; ``ceil((n_iter - burnin) / thin)`` draws are kept
        n_chains : int
            Number of chains; their draws are stacked. Chain ``i`` is seeded
            with ``seed + 1000 * i``
        priors : dict, optional
            Prior specifications for parameters
        verbose : bool
            Whether to print progress
        use_rust : bool
            Whether to use Rust backend when available
        seed : int, optional
            Random seed for reproducibility
        parallel_chains : bool
            Whether to run multiple NumPy chains in spawned worker processes;
            the draws equal those of a serial run. The multithreaded native
            2PL kernel always runs its chains in turn
        """
        n_iter = _validate_count(n_iter, "n_iter", 1)
        burnin = _validate_count(burnin, "burnin", 0)
        thin = _validate_count(thin, "thin", 1)
        if burnin >= n_iter:
            raise MirtValidationError(
                "burnin must be less than n_iter",
                parameter="burnin",
                value=burnin,
                expected=f"< {n_iter}",
            )
        super().__init__(max_iter=n_iter, verbose=verbose)
        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin
        self.n_chains = _validate_count(n_chains, "n_chains", 1)
        self.priors = priors or {}
        self.use_rust = use_rust
        self.seed = seed
        self.parallel_chains = parallel_chains

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        **kwargs: Any,
    ) -> MCMCResult:
        """Fit model using Gibbs sampling.

        Parameters
        ----------
        model : BaseItemModel
            IRT model
        responses : NDArray
            Response matrix

        Returns
        -------
        MCMCResult
            MCMC estimation result with chains and diagnostics

        Raises
        ------
        MirtValidationError
            If ``set_free_parameter_masks`` fixes parameters, which the
            sampler cannot hold.
        """
        from mirt._backend_config import should_use_rust
        from mirt.backends.rust.estimation import gibbs_sample_2pl

        _reject_parameter_restrictions(model, "GibbsSampler")
        responses = self._validate_responses(responses, model.n_items)
        n_persons, n_items = responses.shape

        if should_use_rust(self.use_rust) and _is_2pl_unidimensional(model):
            # The native kernel is multithreaded, so its chains run in turn.
            draws = [
                gibbs_sample_2pl(
                    responses,
                    n_iter=self.n_iter,
                    burnin=self.burnin,
                    thin=self.thin,
                    seed=seed,
                )
                for seed in self._chain_seeds()
            ]
            disc_chain, diff_chain, theta_chain, ll_chain = (
                np.concatenate([np.asarray(chain[k]) for chain in draws], axis=0)
                for k in range(4)
            )

            if not model._parameters:
                model._initialize_parameters()
            model._parameters["discrimination"] = np.mean(disc_chain, axis=0)
            model._parameters["difficulty"] = np.mean(diff_chain, axis=0)
            model._is_fitted = True

            chain_arrays: dict[str, NDArray[np.float64]] = {
                "discrimination": np.asarray(disc_chain),
                "difficulty": np.asarray(diff_chain),
                "theta": np.asarray(theta_chain),
                "log_likelihood": np.asarray(ll_chain),
            }

            rhat = self._compute_rhat(chain_arrays)
            ess = self._compute_ess(chain_arrays)
            ll_mean = float(np.mean(ll_chain))

            dic = self._compute_dic(chain_arrays, model, responses)
            waic = self._compute_waic(chain_arrays, model, responses)

            return MCMCResult(
                model=model,
                chains=chain_arrays,
                log_likelihood=ll_mean,
                dic=dic,
                waic=waic,
                rhat=rhat,
                ess=ess,
                n_iterations=self.n_iter,
                burnin=self.burnin,
                thin=self.thin,
            )

        if not model._parameters:
            model._initialize_parameters()

        if self.n_chains > 1:
            chain_arrays = self._run_chains(model, responses, n_persons)
        else:
            chain_arrays = self._run_single_chain(
                model, responses, n_persons, self.seed
            )

        for name in model.parameters:
            model._parameters[name] = np.mean(chain_arrays[name], axis=0)

        model._is_fitted = True

        rhat = self._compute_rhat(chain_arrays)
        ess = self._compute_ess(chain_arrays)
        ll_mean = float(np.mean(chain_arrays["log_likelihood"]))
        dic = self._compute_dic(chain_arrays, model, responses)
        waic = self._compute_waic(chain_arrays, model, responses)

        return MCMCResult(
            model=model,
            chains=chain_arrays,
            log_likelihood=ll_mean,
            dic=dic,
            waic=waic,
            rhat=rhat,
            ess=ess,
            n_iterations=self.n_iter,
            burnin=self.burnin,
            thin=self.thin,
        )

    def _run_single_chain(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        n_persons: int,
        seed: int | None,
    ) -> dict[str, NDArray]:
        """Run a single MCMC chain."""
        theta = np.zeros((n_persons, model.n_factors))
        rng = np.random.default_rng(seed)

        chains: dict[str, list] = {name: [] for name in model.parameters}
        chains["theta"] = []
        chains["log_likelihood"] = []

        for iteration in range(self.n_iter):
            theta = self._sample_theta_gibbs(model, responses, theta, rng)

            self._sample_parameters(model, responses, theta, rng)

            if iteration >= self.burnin and (iteration - self.burnin) % self.thin == 0:
                for name, values in model.parameters.items():
                    chains[name].append(values.copy())
                chains["theta"].append(theta.copy())
                ll = np.sum(model.log_likelihood(responses, theta))
                chains["log_likelihood"].append(ll)

            if self.verbose and (iteration + 1) % 500 == 0:
                ll = np.sum(model.log_likelihood(responses, theta))
                print(f"Iteration {iteration + 1}/{self.n_iter}: LL = {ll:.4f}")

        return {name: np.array(chain) for name, chain in chains.items()}

    def _run_chains(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        n_persons: int,
    ) -> dict[str, NDArray]:
        """Run ``n_chains`` seeded chains from the same start and stack draws.

        Chain ``i`` uses seed ``seed + 1000 * i``, so worker processes
        (``parallel_chains=True``) reproduce the serial draws exactly. Workers
        are spawned with the parent's backend preference rather than forked
        from a process whose native thread pool may already be running.
        """
        from mirt.utils._parallel import _process_pool, resolve_n_jobs

        seeds = self._chain_seeds()
        if self.parallel_chains:
            try:
                pickle.dumps((self, model))
            except (AttributeError, pickle.PickleError, TypeError) as exc:
                raise MirtValidationError(
                    "parallel chains need a picklable sampler and model; use "
                    "parallel_chains=False for locally defined models",
                    parameter="parallel_chains",
                    value=True,
                ) from exc
            if self.verbose:
                print(f"Running {self.n_chains} chains in parallel...")
            with _process_pool(resolve_n_jobs(-1, self.n_chains)) as executor:
                futures = [
                    executor.submit(
                        self._run_single_chain, model.copy(), responses, n_persons, seed
                    )
                    for seed in seeds
                ]
                all_chains = [future.result() for future in futures]
        else:
            all_chains = [
                self._run_single_chain(model.copy(), responses, n_persons, seed)
                for seed in seeds
            ]

        return {
            name: np.concatenate([chain[name] for chain in all_chains], axis=0)
            for name in all_chains[0]
        }

    def _chain_seeds(self) -> list[int]:
        """Return ``seed + 1000 * i`` for each chain, from a random base if unseeded."""
        base_seed = (
            self.seed
            if self.seed is not None
            else int(np.random.default_rng().integers(0, 2**31))
        )
        return [base_seed + 1000 * i for i in range(self.n_chains)]

    def _sample_theta_gibbs(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample theta using MH within Gibbs."""
        n_persons = theta.shape[0]
        proposal_sd = 0.5

        proposal = theta + rng.normal(0, proposal_sd, theta.shape)

        ll_current = model.log_likelihood(responses, theta)
        ll_proposal = model.log_likelihood(responses, proposal)

        prior_current = stats.norm.logpdf(theta).sum(axis=1)
        prior_proposal = stats.norm.logpdf(proposal).sum(axis=1)

        log_alpha = (ll_proposal + prior_proposal) - (ll_current + prior_current)
        accept = np.log(rng.random(n_persons)) < log_alpha

        return np.where(accept[:, None], proposal, theta)

    def _sample_parameters(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> None:
        """Sample item parameters using MH."""
        proposal_sd = 0.1

        for name, values in model.parameters.items():
            proposal = values + rng.normal(0, proposal_sd, values.shape)

            if "discrimination" in name or "slope" in name:
                proposal = np.clip(proposal, 0.1, 5.0)
            elif "difficulty" in name or "intercept" in name:
                proposal = np.clip(proposal, -6.0, 6.0)

            model._parameters[name] = proposal
            ll_proposal = np.sum(model.log_likelihood(responses, theta))

            model._parameters[name] = values
            ll_current = np.sum(model.log_likelihood(responses, theta))

            log_alpha = ll_proposal - ll_current

            if np.log(rng.random()) < log_alpha:
                model._parameters[name] = proposal

    def _compute_rhat(self, chains: dict[str, NDArray]) -> dict[str, float]:
        """Compute Gelman-Rubin R-hat diagnostic."""
        rhat = {}

        for name, chain in chains.items():
            if name in ("theta", "log_likelihood"):
                continue

            if chain.ndim == 1:
                values = chain
            else:
                values = chain.mean(axis=tuple(range(1, chain.ndim)))

            n = len(values)
            if n < 10:
                rhat[name] = np.nan
                continue

            first_half = values[: n // 2]
            second_half = values[n // 2 :]

            B = (n // 2) * np.var([first_half.mean(), second_half.mean()])
            W = (np.var(first_half) + np.var(second_half)) / 2

            if W > 0:
                var_est = (1 - 1 / (n // 2)) * W + B / (n // 2)
                rhat[name] = float(np.sqrt(var_est / W))
            else:
                rhat[name] = 1.0

        return rhat

    def _compute_ess(self, chains: dict[str, NDArray]) -> dict[str, float]:
        """Compute effective sample size."""
        ess = {}

        for name, chain in chains.items():
            if name in ("theta", "log_likelihood"):
                continue

            if chain.ndim == 1:
                values = chain
            else:
                values = chain.mean(axis=tuple(range(1, chain.ndim)))

            n = len(values)
            if n < 10:
                ess[name] = float(n)
                continue

            acf = np.correlate(
                values - values.mean(), values - values.mean(), mode="full"
            )
            acf = acf[n - 1 :] / acf[n - 1]

            neg_idx = np.where(acf < 0)[0]
            if len(neg_idx) > 0:
                cutoff = neg_idx[0]
            else:
                cutoff = min(n // 2, 100)

            tau = 1 + 2 * np.sum(acf[1:cutoff])
            ess[name] = float(n / max(tau, 1))

        return ess

    def _compute_dic(
        self,
        chains: dict[str, NDArray],
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> float:
        """Compute Deviance Information Criterion."""
        ll_mean = np.mean(chains["log_likelihood"])
        deviance_mean = -2 * ll_mean

        theta_mean = np.mean(chains["theta"], axis=0)
        ll_at_mean = np.sum(model.log_likelihood(responses, theta_mean))
        deviance_at_mean = -2 * ll_at_mean

        pd = deviance_mean - deviance_at_mean

        return float(deviance_mean + pd)

    def _compute_waic(
        self,
        chains: dict[str, NDArray],
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> float:
        """Compute stable pointwise WAIC from paired posterior draws."""
        theta_chain = np.asarray(chains["theta"], dtype=np.float64)
        if theta_chain.ndim != 3 or theta_chain.shape[0] == 0:
            raise MirtEstimationError(
                "theta chain must have shape (n_samples, n_persons, n_factors)"
            )

        n_samples, n_chain_persons, n_chain_factors = theta_chain.shape
        n_persons = responses.shape[0]
        if n_chain_persons != n_persons or n_chain_factors != model.n_factors:
            raise MirtEstimationError(
                "theta chain dimensions must match the fitted data and model",
                expected=(n_persons, model.n_factors),
                actual=(n_chain_persons, n_chain_factors),
            )
        if not np.all(np.isfinite(theta_chain)):
            raise MirtEstimationError("theta chain must contain only finite values")

        parameter_chains: dict[str, NDArray[np.float64]] = {}
        for name, parameter in model.parameters.items():
            if name not in chains:
                raise MirtEstimationError(
                    "posterior parameter chain is missing",
                    parameter=name,
                )
            values = np.asarray(chains[name], dtype=np.float64)
            expected_shape = (n_samples, *parameter.shape)
            if values.shape != expected_shape:
                raise MirtEstimationError(
                    "posterior parameter chain has an unexpected shape",
                    parameter=name,
                    expected=expected_shape,
                    actual=values.shape,
                )
            if not np.all(np.isfinite(values)):
                raise MirtEstimationError(
                    "posterior parameter chain must contain only finite values",
                    parameter=name,
                )
            parameter_chains[name] = values

        evaluation_model = model.copy()
        evaluation_model._is_fitted = True
        log_sum_exp = np.full(n_persons, -np.inf, dtype=np.float64)
        running_mean = np.zeros(n_persons, dtype=np.float64)
        running_m2 = np.zeros(n_persons, dtype=np.float64)

        for sample_index in range(n_samples):
            for name, values in parameter_chains.items():
                evaluation_model._parameters[name] = values[sample_index]

            pointwise = np.asarray(
                evaluation_model.log_likelihood(
                    responses,
                    theta_chain[sample_index],
                ),
                dtype=np.float64,
            )
            if pointwise.shape != (n_persons,) or not np.all(np.isfinite(pointwise)):
                raise MirtEstimationError(
                    "model returned invalid pointwise posterior log-likelihoods",
                    iteration=sample_index,
                    expected=(n_persons,),
                    actual=pointwise.shape,
                )

            count = sample_index + 1
            delta = pointwise - running_mean
            running_mean += delta / count
            running_m2 += delta * (pointwise - running_mean)
            log_sum_exp = np.logaddexp(log_sum_exp, pointwise)

        log_pointwise_predictive_density = np.sum(
            log_sum_exp - np.log(float(n_samples))
        )
        effective_parameters = np.sum(running_m2 / n_samples)
        waic = -2.0 * (log_pointwise_predictive_density - effective_parameters)
        if not np.isfinite(waic):
            raise MirtEstimationError("WAIC is non-finite")
        return float(waic)
