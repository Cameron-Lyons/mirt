"""MCMC and MHRM Estimation for IRT Models.

This module provides stochastic estimation methods:
- MHRM (Metropolis-Hastings Robbins-Monro)
- Gibbs Sampling for full Bayesian inference

Uses fast Rust backend when available for 2PL models.
"""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtEstimationError, MirtValidationError

if TYPE_CHECKING:
    from mirt.estimation.priors import Prior, PriorSpecification
    from mirt.models.base import BaseItemModel

from mirt.estimation._graded_order import (
    THRESHOLD_GAP,
    graded_order,
    graded_threshold_constraint,
)
from mirt.estimation.base import BaseEstimator
from mirt.results.fit_result import FitResult

PosteriorValue = NDArray[np.float64] | np.float64
PosteriorSummary = dict[str, dict[str, PosteriorValue]]
CredibleIntervals = dict[str, tuple[PosteriorValue, PosteriorValue]]
ChainDiagnostics = dict[str, dict[str, NDArray[np.float64]]]

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
# MH-RM stops once the largest change of a free coordinate stays below tol
# for this many consecutive post-burn-in cycles, as R mirt does.
_MHRM_STOP_CYCLES = 3

# Convergence thresholds of Vehtari et al. (2021): a rank-normalized split
# R-hat below 1.01 and bulk and tail effective sample sizes of at least 100
# per chain for every free item coordinate.
_RHAT_THRESHOLD = 1.01
_MIN_ESS_PER_CHAIN = 100
# Chains that are not item parameters.
_NON_ITEM_CHAINS = ("theta", "log_likelihood")

# Marginal log-likelihoods of models with up to this many factors integrate
# on EM's default grid; larger models use importance sampling.
_MAX_QUADRATURE_FACTORS = 3
_MAX_GRID_ENTRIES = 4_000_000
_IMPORTANCE_DRAWS = 3000
_IMPORTANCE_BLOCK = 250
# Share of importance draws taken from the latent density, which bounds the
# weights, and the variance inflation of the posterior approximations.
_DEFENSIVE_SHARE = 0.1
_POSTERIOR_INFLATION = 1.5

# Sampling scales of item coordinates (see _ItemBlocks).
_IDENTITY, _LOWER_LOG, _LOGIT, _GAP = range(4)
# Initial random-walk standard deviations on the sampling scale.
_GIBBS_ITEM_STEP = 0.1
_GIBBS_LOGIT_STEP = 0.5
_GIBBS_THETA_STEP = 0.5
# Robbins-Monro exponent of the burn-in proposal adaptation.
_ADAPTATION_DECAY = 0.6
# Spread, on the sampling scale, of the starting points of several chains.
_DISPERSION_SD = 0.5


def _validate_count(value: int, name: str, minimum: int) -> int:
    """Return an integer sampler control of at least ``minimum``."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        expected = {0: "non-negative integer", 1: "positive integer"}.get(
            minimum, f"integer of at least {minimum}"
        )
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


def _validate_latent_prior(
    prior_mean: Any,
    prior_cov: Any,
    n_factors: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a finite latent mean and a positive-definite latent covariance."""
    try:
        mean = (
            np.zeros(n_factors)
            if prior_mean is None
            else np.array(prior_mean, dtype=np.float64)
        )
        cov = (
            np.eye(n_factors)
            if prior_cov is None
            else np.array(prior_cov, dtype=np.float64)
        )
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            "prior_mean and prior_cov must be numeric arrays",
            parameter="prior_mean",
        ) from exc
    if mean.shape != (n_factors,) or not np.all(np.isfinite(mean)):
        raise MirtValidationError(
            "prior_mean must be a finite vector with one entry per factor",
            parameter="prior_mean",
            value=mean.shape,
            expected=f"({n_factors},)",
        )
    if (
        cov.shape != (n_factors, n_factors)
        or not np.all(np.isfinite(cov))
        or not np.allclose(cov, cov.T)
    ):
        raise MirtValidationError(
            "prior_cov must be a finite symmetric matrix with one row per factor",
            parameter="prior_cov",
            value=cov.shape,
            expected=f"({n_factors}, {n_factors})",
        )
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise MirtValidationError(
            "prior_cov must be positive definite",
            parameter="prior_cov",
        ) from exc
    return mean, 0.5 * (cov + cov.T)


def _is_standard_normal(mean: NDArray[np.float64], cov: NDArray[np.float64]) -> bool:
    """Whether a latent density is the standard normal."""
    return bool(np.all(mean == 0.0) and np.array_equal(cov, np.eye(len(mean))))


# ---------------------------------------------------------------------------
# Convergence diagnostics (Vehtari, Gelman, Simpson, Carpenter and Buerkner,
# 2021).
# ---------------------------------------------------------------------------


def _split_draws(draws: NDArray[np.float64]) -> NDArray[np.float64]:
    """Split every chain of ``(n_chains, n_draws, P)`` draws into halves.

    The middle draw of an odd-length chain is dropped.
    """
    half = draws.shape[1] // 2
    return np.concatenate((draws[:, :half], draws[:, draws.shape[1] - half :]))


def _rank_normalize(draws: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return normal scores of the ranks of each column over all chains."""
    n_chains, n_draws, n_values = draws.shape
    total = n_chains * n_draws
    ranks = stats.rankdata(draws.reshape(total, n_values), axis=0)
    return special.ndtri((ranks - 0.375) / (total + 0.25)).reshape(draws.shape)


def _basic_rhat(split: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the potential scale reduction of each column of split chains."""
    n_draws = split.shape[1]
    within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    between = n_draws * np.var(np.mean(split, axis=1), axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt((between / within + n_draws - 1) / n_draws)
    # Chains that are each constant disagree without bound, or not at all.
    flat = ~(within > 0.0)
    rhat[flat] = np.where(between[flat] > 0.0, np.inf, np.nan)
    return rhat


def _effective_sample_size(split: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the effective sample size of each column of split chains.

    The multi-chain autocorrelations are truncated and made monotone by
    Geyer's initial monotone sequence, as Stan and the R package posterior
    do; the estimate is at most ``S log10(S)`` for ``S`` draws.
    """
    n_chains, n_draws, n_values = split.shape
    total = n_chains * n_draws
    centered = split - np.mean(split, axis=1, keepdims=True)
    size = 1 << (2 * n_draws - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=size, axis=1)
    autocovariance = (
        np.fft.irfft(spectrum * np.conjugate(spectrum), n=size, axis=1)[:, :n_draws]
        / n_draws
    )
    mean_variance = np.mean(autocovariance[:, 0], axis=0) * n_draws / (n_draws - 1)
    variance = mean_variance * (n_draws - 1) / n_draws + np.var(
        np.mean(split, axis=1), axis=0, ddof=1
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = 1.0 - (mean_variance - np.mean(autocovariance, axis=0)) / variance
    kept = np.zeros_like(rho)
    kept[0] = 1.0
    kept[1] = rho[1]
    columns = np.arange(n_values)
    last = np.zeros(n_values, dtype=np.intp)
    last_even = np.ones(n_values)
    active = np.isfinite(rho[1]) & (1.0 + rho[1] > 0.0)
    lag = 0
    # Sum autocorrelation pairs while they stay positive.
    while lag < n_draws - 5 and active.any():
        lag += 2
        even, odd = rho[lag], rho[lag + 1]
        pair = even + odd
        keep = active & (pair >= 0.0)
        kept[lag, keep] = even[keep]
        kept[lag + 1, keep] = odd[keep]
        last[active] = lag
        last_even[active] = even[active]
        active &= np.isfinite(pair) & (pair > 0.0)
    positive = last_even > 0.0
    kept[last[positive], columns[positive]] = last_even[positive]
    for lag in range(2, int(last.max(initial=0)) - 1, 2):
        previous = kept[lag - 2] + kept[lag - 1]
        lower = (lag <= last - 2) & (kept[lag] + kept[lag + 1] > previous)
        kept[lag, lower] = kept[lag + 1, lower] = previous[lower] / 2.0
    lags = np.arange(n_draws)[:, None]
    tau = (
        -1.0
        + 2.0 * np.sum(np.where(lags < last, kept, 0.0), axis=0)
        + kept[last, columns]
    )
    ess = total / np.maximum(tau, 1.0 / np.log10(total))
    ess[~(variance > 0.0)] = np.nan
    return ess


def _convergence_statistics(
    draws: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return rank-normalized split R-hat and bulk and tail ESS per column.

    ``draws`` has shape ``(n_chains, n_draws, P)``. R-hat is the larger of
    the rank-normalized split R-hat of the draws and of their distances from
    the median; tail ESS is the smaller ESS of the 5% and 95% quantile
    indicators (Vehtari et al., 2021). Columns that never change, and every
    column of chains shorter than four draws, are ``NaN``.
    """
    n_values = draws.shape[2]
    if draws.shape[1] < 4:
        empty = np.full(n_values, np.nan)
        return empty, empty.copy(), empty.copy()
    split = _split_draws(draws)
    folded = np.abs(split - np.median(split, axis=(0, 1)))
    rhat = np.fmax(
        _basic_rhat(_rank_normalize(split)), _basic_rhat(_rank_normalize(folded))
    )
    ess_bulk = _effective_sample_size(_rank_normalize(split))
    pooled = draws.reshape(-1, n_values)
    tails = [
        _effective_sample_size(
            (split <= np.quantile(pooled, probability, axis=0)).astype(np.float64)
        )
        for probability in (0.05, 0.95)
    ]
    ess_tail = np.fmin(*tails)
    constant = np.ptp(pooled, axis=0) == 0.0
    rhat[constant] = ess_bulk[constant] = ess_tail[constant] = np.nan
    return rhat, ess_bulk, ess_tail


def _diagnose_chains(
    chains: Mapping[str, NDArray[np.float64]],
    model: BaseItemModel,
    n_chains: int,
) -> tuple[ChainDiagnostics, dict[str, float], dict[str, float], dict[str, float]]:
    """Return per-coordinate diagnostics and each parameter's worst values.

    Stacked draws are split into ``n_chains`` chains of equal length. Only
    the coordinates that ``model.free_parameter_masks`` leaves free are
    assessed; fixed ones are ``NaN`` and do not enter the extremes, while a
    free coordinate that never moved makes them ``NaN``.
    """
    try:
        masks = model.free_parameter_masks
    except AttributeError:
        masks = {}
    per_coordinate: ChainDiagnostics = {}
    worst_rhat: dict[str, float] = {}
    worst_bulk: dict[str, float] = {}
    worst_tail: dict[str, float] = {}
    for name, chain in chains.items():
        if name in _NON_ITEM_CHAINS:
            continue
        values = np.asarray(chain, dtype=np.float64)
        if values.ndim == 0 or values.shape[0] % n_chains:
            raise ValueError(
                f"posterior chain '{name}' must hold the same number of draws "
                "for every chain"
            )
        shape = values.shape[1:]
        draws = values.reshape(n_chains, values.shape[0] // n_chains, -1)
        mask = masks.get(name)
        free = (
            np.ones(draws.shape[2], dtype=np.bool_)
            if mask is None or np.shape(mask) != shape
            else np.asarray(mask, dtype=np.bool_).reshape(-1)
        )
        statistics = [np.full(draws.shape[2], np.nan) for _ in range(3)]
        if free.any():
            for array, values_free in zip(
                statistics, _convergence_statistics(draws[:, :, free]), strict=True
            ):
                array[free] = values_free
            worst_rhat[name] = float(np.max(statistics[0][free]))
            worst_bulk[name] = float(np.min(statistics[1][free]))
            worst_tail[name] = float(np.min(statistics[2][free]))
        per_coordinate[name] = {
            key: array.reshape(shape)
            for key, array in zip(
                ("rhat", "ess_bulk", "ess_tail"), statistics, strict=True
            )
        }
    return per_coordinate, worst_rhat, worst_bulk, worst_tail


# ---------------------------------------------------------------------------
# Marginal log-likelihood
# ---------------------------------------------------------------------------


def _gaussian_log_density(
    points: NDArray[np.float64],
    center: NDArray[np.float64],
    precision: NDArray[np.float64],
    log_determinant: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Return normal log-densities of ``(n, s, F)`` points.

    ``center`` is ``(F,)`` or ``(n, F)``, ``precision`` ``(F, F)`` or
    ``(n, F, F)`` and ``log_determinant`` the log-determinant of the
    covariance, a scalar or ``(n,)``.
    """
    n_factors = points.shape[-1]
    offset = points - (center if center.ndim == 1 else center[:, None, :])
    if precision.ndim == 2:
        quadratic = np.einsum("nsi,ij,nsj->ns", offset, precision, offset)
        log_det = np.asarray(log_determinant, dtype=np.float64)
    else:
        quadratic = np.einsum("nsi,nij,nsj->ns", offset, precision, offset)
        log_det = np.asarray(log_determinant, dtype=np.float64)[:, None]
    return -0.5 * (n_factors * np.log(2.0 * np.pi) + log_det + quadratic)


def _theta_moments(
    theta_draws: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return each person's mean and covariance over ``(S, N, F)`` draws."""
    draws = np.asarray(theta_draws, dtype=np.float64)
    means = np.mean(draws, axis=0)
    centered = draws - means
    covariances = np.einsum("snf,sng->nfg", centered, centered) / max(
        draws.shape[0] - 1, 1
    )
    return means, covariances


def _marginal_log_likelihood(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    mean: NDArray[np.float64] | None = None,
    cov: NDArray[np.float64] | None = None,
    posterior: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Return the marginal log-likelihood under a ``N(mean, cov)`` density.

    Models with up to three factors integrate on EM's default grid (61, 31
    or 15 points per dimension, see
    :func:`~mirt.estimation.quadrature.build_quadrature`), mapped to the
    latent density, so the value equals EM's log-likelihood at the same
    parameters. Larger models use importance sampling: each person's draws
    come from a normal approximation of the ability posterior, given as
    ``posterior`` means and covariances, mixed with the latent density so
    the weights stay bounded; without ``posterior`` the draws come from the
    latent density alone.
    """
    from mirt.estimation._mc_likelihood import sampled_log_likelihoods
    from mirt.estimation._patterns import (
        compress_responses,
        supports_pattern_compression,
    )
    from mirt.estimation.quadrature import build_quadrature, resolve_quadpts

    n_factors = model.n_factors
    mean = np.zeros(n_factors) if mean is None else np.asarray(mean, np.float64)
    cov = np.eye(n_factors) if cov is None else np.asarray(cov, np.float64)
    factor = np.linalg.cholesky(cov)

    if n_factors <= _MAX_QUADRATURE_FACTORS:
        quadrature = build_quadrature(resolve_quadpts(None, n_factors), n_factors)
        nodes = quadrature.nodes @ factor.T + mean
        log_weights = np.log(quadrature.weights)
        frequencies = None
        if supports_pattern_compression(model):
            responses, frequencies = compress_responses(responses)
        rows = max(1, _MAX_GRID_ENTRIES // len(nodes))
        person = np.empty(len(responses))
        for start in range(0, len(responses), rows):
            joint = np.asarray(
                model.log_likelihood_batch(responses[start : start + rows], nodes),
                dtype=np.float64,
            )
            person[start : start + rows] = special.logsumexp(
                joint + log_weights, axis=1
            )
        if not np.all(np.isfinite(person)):
            raise MirtEstimationError("marginal log-likelihood is non-finite")
        return float(np.sum(person) if frequencies is None else person @ frequencies)

    rng = np.random.default_rng() if rng is None else rng
    n_persons = responses.shape[0]
    precision = np.linalg.inv(cov)
    log_det = float(2.0 * np.sum(np.log(np.diag(factor))))
    share = 1.0
    if posterior is not None:
        share = _DEFENSIVE_SHARE
        centers = np.asarray(posterior[0], dtype=np.float64)
        spread = _POSTERIOR_INFLATION * np.asarray(posterior[1], dtype=np.float64)
        spread = spread + 1e-6 * np.eye(n_factors)
        person_factors = np.linalg.cholesky(spread)
        person_precision = np.linalg.inv(spread)
        person_log_det = 2.0 * np.sum(
            np.log(np.diagonal(person_factors, axis1=1, axis2=2)), axis=1
        )
    total = np.full(n_persons, -np.inf)
    drawn = 0
    while drawn < _IMPORTANCE_DRAWS:
        size = min(_IMPORTANCE_BLOCK, _IMPORTANCE_DRAWS - drawn)
        noise = rng.standard_normal((n_persons, size, n_factors))
        points = mean + noise @ factor.T
        log_prior = _gaussian_log_density(points, mean, precision, log_det)
        log_proposal = log_prior
        if posterior is not None:
            approximate = centers[:, None, :] + np.einsum(
                "nij,nsj->nsi", person_factors, noise
            )
            from_prior = rng.random((n_persons, size)) < share
            points = np.where(from_prior[..., None], points, approximate)
            log_prior = _gaussian_log_density(points, mean, precision, log_det)
            log_proposal = np.logaddexp(
                np.log(share) + log_prior,
                np.log1p(-share)
                + _gaussian_log_density(
                    points, centers, person_precision, person_log_det
                ),
            )
        conditional = sampled_log_likelihoods(model, responses, points)
        if conditional is None:
            conditional = np.column_stack(
                [model.log_likelihood(responses, points[:, s]) for s in range(size)]
            )
        total = np.logaddexp(
            total,
            special.logsumexp(conditional + log_prior - log_proposal, axis=1),
        )
        drawn += size
    person = total - np.log(_IMPORTANCE_DRAWS)
    if not np.all(np.isfinite(person)):
        raise MirtEstimationError("marginal log-likelihood is non-finite")
    return float(np.sum(person))


@dataclass
class MCMCResult:
    """Result from MCMC estimation.

    Attributes
    ----------
    model : BaseItemModel
        Fitted model with posterior mean parameters
    chains : dict
        Posterior draws of each item parameter, of ``theta`` and of the
        conditional ``log_likelihood`` of the responses given each draw. The
        draws of ``n_chains`` chains are stacked along the first axis, chain
        after chain.
    log_likelihood : float
        Marginal log-likelihood at the posterior-mean item parameters,
        integrated over the latent density like EM's (see
        :class:`GibbsSampler`), so AIC, BIC and likelihood-ratio tests built
        from it compare with EM fits
    dic : float
        Deviance Information Criterion, from the conditional log-likelihood
        of the draws
    waic : float
        Watanabe-Akaike Information Criterion
    rhat : dict
        Largest rank-normalized split R-hat (Vehtari et al., 2021) over the
        free coordinates of each item parameter, computed within the
        ``n_chains`` chains
    ess : dict
        Smallest bulk effective sample size of each item parameter
    n_iterations, burnin, thin : int
        Sampling schedule of every chain
    n_chains : int
        Number of chains stacked in ``chains``
    ess_tail : dict
        Smallest tail effective sample size of each item parameter, the
        smaller of the ESS of its 5% and 95% quantiles

    References
    ----------
    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Buerkner, P.-C.
    (2021). Rank-normalization, folding, and localization: An improved
    R-hat for assessing convergence of MCMC. Bayesian Analysis, 16(2),
    667-718.
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
    n_chains: int = 1
    ess_tail: dict[str, float] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        """Whether every item parameter passes the convergence checks.

        Every free coordinate needs a rank-normalized split R-hat below
        1.01 and bulk and tail effective sample sizes of at least 100 per
        chain, the thresholds Vehtari et al. (2021) recommend. Undefined
        diagnostics, such as those of a coordinate that never moved, fail.
        """
        if not self.rhat or not self.ess:
            return False
        minimum = _MIN_ESS_PER_CHAIN * max(int(self.n_chains), 1)
        if not all(
            np.isfinite(value) and value < _RHAT_THRESHOLD
            for value in self.rhat.values()
        ):
            return False
        return all(
            np.isfinite(value) and value >= minimum
            for value in (*self.ess.values(), *self.ess_tail.values())
        )

    def convergence_diagnostics(self) -> ChainDiagnostics:
        """Return R-hat and bulk and tail ESS of every item-parameter coordinate.

        Returns
        -------
        dict
            For each item-parameter chain, ``"rhat"``, ``"ess_bulk"`` and
            ``"ess_tail"`` arrays shaped like the parameter, computed within
            the ``n_chains`` stacked chains. Coordinates the model holds
            fixed are ``NaN``.
        """
        return _diagnose_chains(self.chains, self.model, self.n_chains)[0]

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
            f"Chains: {self.n_chains}",
            "",
            f"Log-likelihood: {self.log_likelihood:.4f}",
            f"DIC: {self.dic:.4f}",
            f"WAIC: {self.waic:.4f}",
            "",
            "Convergence (largest rank-normalized split R-hat):",
        ]

        for name, rhat in self.rhat.items():
            status = "OK" if rhat < _RHAT_THRESHOLD else "WARNING"
            lines.append(f"  {name}: {rhat:.4f} ({status})")

        if self.ess:
            lines.extend(("", "Effective sample size:"))
            for name, ess in self.ess.items():
                lines.append(f"  {name}: {ess:.1f}")
        if self.ess_tail:
            lines.extend(("", "Tail effective sample size:"))
            for name, ess in self.ess_tail.items():
                lines.append(f"  {name}: {ess:.1f}")

        minimum = _MIN_ESS_PER_CHAIN * max(int(self.n_chains), 1)
        lines.extend(
            (
                "",
                f"Converged: {'yes' if self.converged else 'no'} (R-hat < "
                f"{_RHAT_THRESHOLD}, ESS >= {minimum})",
            )
        )
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
    later gains decrease (see ``gain_sequence``) and the estimates are the
    final iterate (Cai, 2010). Items are updated blockwise, so every built-in
    item family, including polytomous and multidimensional ones, is
    supported except the rating scale models (RSM and GRSM), whose
    thresholds are shared by all items. Coordinates fixed with
    ``set_free_parameter_masks`` keep their values, so confirmatory loading
    patterns can be fitted.

    The fit stops, with ``converged=True``, once no free coordinate has
    changed by ``tol`` or more in three consecutive post-burn-in cycles, the
    rule of R mirt. A fit that reaches ``n_cycles`` first reports
    ``converged=False`` and warns. The reported log-likelihood, and the AIC
    and BIC built from it, is the marginal log-likelihood at the estimates,
    integrated as EM integrates it for up to three factors and by importance
    sampling beyond (see :class:`GibbsSampler`).

    Standard errors of unidimensional 1PL-4PL, GRM, GPCM and PCM fits come
    from the exact observed information of the marginal likelihood at the
    estimates (Louis, 1982), integrated by Gauss-Hermite quadrature over the
    ability density; they match EM's ``se_method="oakes"`` and fill
    ``FitResult.vcov``. Other models report the standard deviation of the
    post-burn-in iterates, labelled ``se_method="mhrm_iterate_sd"``. That
    measures the Robbins-Monro noise around the estimate, not sampling
    variability, so it is no substitute for a standard error.

    The fast native kernel runs the burn-in cycles of unidimensional 2PL
    fits with one chain, the standard normal ability density and no fixed
    coordinates; the decreasing-gain cycles always run in NumPy.

    References
    ----------
    Cai, L. (2010). Metropolis-Hastings Robbins-Monro algorithm for
    confirmatory item factor analysis. Journal of Educational and
    Behavioral Statistics, 35(3), 307-335.

    Louis, T. A. (1982). Finding the observed information matrix when using
    the EM algorithm. Journal of the Royal Statistical Society: Series B,
    44(2), 226-233.
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
        compute_standard_errors: bool = True,
        n_quadpts: int = 21,
        tol: float = 1e-3,
    ) -> None:
        """Initialize MHRM estimator.

        Parameters
        ----------
        n_cycles : int
            Largest number of MHRM cycles, burn-in included
        burnin : int
            Number of initial unit-gain cycles, which are excluded from the
            stopping rule and from the spread of iterates that serves as
            standard errors for families without exact derivatives
        n_chains : int
            Number of independent ability chains imputed in every cycle. The
            complete-data score and information average over them (Cai,
            2010), which lowers the Robbins-Monro noise of each step.
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
        compute_standard_errors : bool
            Whether to compute standard errors. Without them the result has
            an empty standard-error mapping.
        n_quadpts : int
            Gauss-Hermite points for the observed-information standard errors.
        tol : float
            Stopping tolerance: the fit stops once the largest change of a
            free coordinate stays below ``tol`` for three consecutive
            post-burn-in cycles. R mirt's MH-RM default is the same 0.001. A
            smaller value runs longer under smaller gains, which lowers the
            Monte Carlo error of the estimates.
        """
        n_cycles = _validate_count(n_cycles, "n_cycles", 1)
        burnin = _validate_count(burnin, "burnin", 0)
        n_chains = _validate_count(n_chains, "n_chains", 1)
        n_quadpts = _validate_count(n_quadpts, "n_quadpts", 2)
        if not isinstance(compute_standard_errors, (bool, np.bool_)):
            raise MirtValidationError(
                "compute_standard_errors must be a boolean",
                parameter="compute_standard_errors",
                value=compute_standard_errors,
                expected="bool",
            )
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
        super().__init__(max_iter=n_cycles, tol=tol, verbose=verbose)
        self.n_cycles = n_cycles
        self.burnin = burnin
        self.n_chains = n_chains
        self.proposal_sd = float(proposal_sd)
        self.gain_sequence = gain_sequence
        self.use_rust = use_rust
        self.seed = seed
        self.compute_standard_errors = bool(compute_standard_errors)
        self.n_quadpts = n_quadpts

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
    ) -> FitResult:
        """Fit model using MHRM algorithm.

        Parameters
        ----------
        model : BaseItemModel
            IRT model to fit. Coordinates fixed with
            ``set_free_parameter_masks`` keep their values.
        responses : NDArray
            Response matrix (n_persons, n_items)
        prior_mean, prior_cov : ndarray, optional
            Mean and covariance of the normal ability density, standard
            normal by default. A non-standard density sets the scale of the
            items, as in :meth:`~mirt.estimation.em.EMEstimator.fit`, and is
            recorded as the result's ``latent_mean`` and
            ``latent_covariance``.

        Returns
        -------
        FitResult
            Fitted model result. ``log_likelihood`` is the marginal
            log-likelihood at the estimates, ``n_iterations`` the number of
            cycles run and ``converged`` whether the stopping rule held.
            ``se_method`` records how the standard errors were computed (see
            the class notes).

        Raises
        ------
        MirtValidationError
            If ``prior_mean`` or ``prior_cov`` do not describe a normal
            density of the model's factors.

        Warns
        -----
        RuntimeWarning
            If ``n_cycles`` ran out before the stopping rule held.
        """
        from mirt._backend_config import should_use_rust
        from mirt.backends.rust.estimation import mhrm_fit_2pl
        from mirt.estimation.base import _parameter_bounds

        responses = self._validate_responses(responses, model.n_items, model)
        n_persons = responses.shape[0]
        mean, cov = _validate_latent_prior(prior_mean, prior_cov, model.n_factors)
        standard = _is_standard_normal(mean, cov)
        if not model._parameters:
            model._initialize_parameters()
        self._check_shared_parameters(model)
        seed = (
            int(self.seed)
            if self.seed is not None
            else int(np.random.default_rng().integers(0, 2**31))
        )

        first_cycle = 0
        if (
            should_use_rust(self.use_rust)
            and _is_2pl_unidimensional(model)
            and not model._free_parameter_restrictions
            and standard
            and self.n_chains == 1
            and self.burnin > 0
        ):
            # The native kernel returns its final iterate when every cycle is
            # a burn-in cycle; the decreasing-gain cycles continue below.
            first_cycle = min(self.burnin, self.n_cycles)
            discrimination, difficulty, _ = mhrm_fit_2pl(
                responses,
                n_cycles=first_cycle,
                burnin=first_cycle,
                proposal_sd=self.proposal_sd,
                seed=seed,
                gain_sequence=self.gain_sequence,
                # The NumPy cycles that follow keep EM's boxes.
                disc_bounds=_parameter_bounds(model, "discrimination"),
                diff_bounds=_parameter_bounds(model, "difficulty"),
            )
            model.set_parameters(
                discrimination=np.asarray(discrimination, dtype=np.float64),
                difficulty=np.asarray(difficulty, dtype=np.float64),
            )

        rng = np.random.default_rng(seed)
        n_iterations, converged, history, moments = self._run_cycles(
            model, responses, rng, mean, cov, first_cycle
        )
        # The estimates are the final iterate, as in Cai (2010): under the
        # decreasing gains it is closer to the maximum-likelihood estimates
        # than the average of the post-burn-in iterates.
        model._is_fitted = True

        if not converged:
            reason = (
                "no cycles followed the burn-in"
                if self.burnin >= self.n_cycles
                else "the largest change of a free coordinate did not stay "
                f"below tol={self.tol:g} for {_MHRM_STOP_CYCLES} consecutive "
                "cycles"
            )
            warnings.warn(
                f"MHRM stopped at n_cycles={self.n_cycles} before converging: "
                f"{reason}; increase n_cycles",
                RuntimeWarning,
                stacklevel=2,
            )

        log_likelihood = _marginal_log_likelihood(
            model, responses, mean, cov, posterior=moments, rng=rng
        )
        standard_errors, se_method, covariance = self._standard_errors(
            model, responses, history, mean, cov
        )
        n_params = model.n_parameters
        return FitResult(
            model=model,
            log_likelihood=log_likelihood,
            n_iterations=n_iterations,
            converged=converged,
            standard_errors=standard_errors,
            aic=-2 * log_likelihood + 2 * n_params,
            bic=-2 * log_likelihood + np.log(n_persons) * n_params,
            n_observations=n_persons,
            n_parameters=n_params,
            se_method=se_method,
            vcov=covariance,
            latent_covariance=None
            if np.array_equal(cov, np.eye(model.n_factors))
            else cov,
            latent_mean=None if np.all(mean == 0.0) else mean,
        )

    def _run_cycles(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        rng: np.random.Generator,
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
        first_cycle: int,
    ) -> tuple[
        int,
        bool,
        dict[str, list[NDArray[np.float64]]],
        tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    ]:
        """Run the cycles from ``first_cycle`` until the stopping rule holds.

        Returns the number of cycles run in total, whether the stopping rule
        held, the post-burn-in iterates of every parameter and, for models
        whose marginal likelihood needs importance sampling, the post-burn-in
        mean and covariance of each person's ability draws.
        """
        n_persons, n_factors = responses.shape[0], model.n_factors
        n_chains = self.n_chains
        # Several chains impute stacked copies of the data, each weighted
        # 1 / n_chains, so the item steps average their complete-data terms.
        data = responses if n_chains == 1 else np.tile(responses, (n_chains, 1))
        weights = None if n_chains == 1 else np.full(len(data), 1.0 / n_chains)
        precision = None if _is_standard_normal(mean, cov) else np.linalg.inv(cov)
        theta = np.zeros((len(data), n_factors)) + mean
        # Let the ability chain leave its degenerate start before the first
        # unit-gain step, whose complete-data information needs spread draws.
        for _ in range(_MHRM_WARMUP_SWEEPS):
            theta = self._sample_theta(model, data, theta, rng, mean, precision)

        information: list[NDArray[np.float64] | None] = [None] * model.n_items
        history: dict[str, list[NDArray[np.float64]]] = {
            name: [] for name in model.parameters
        }
        track = n_factors > _MAX_QUADRATURE_FACTORS
        theta_sum = np.zeros((n_persons, n_factors))
        theta_outer = np.zeros((n_persons, n_factors, n_factors))
        previous = model.parameters
        quiet = 0
        n_iterations = first_cycle
        converged = False
        for cycle in range(first_cycle, self.n_cycles):
            theta = self._sample_theta(model, data, theta, rng, mean, precision)
            gain = self._compute_gain(cycle)
            self._update_parameters(model, data, theta, gain, information, weights)
            n_iterations = cycle + 1
            current = model.parameters

            if cycle >= self.burnin:
                for name, values in current.items():
                    history[name].append(values)
                if track:
                    draws = theta[:n_persons]
                    theta_sum += draws
                    theta_outer += draws[:, :, None] * draws[:, None, :]
                quiet = (
                    quiet + 1 if _largest_change(previous, current) < self.tol else 0
                )
                if quiet >= _MHRM_STOP_CYCLES:
                    converged = True
                    break
            previous = current

            if self.verbose and (cycle + 1) % 100 == 0:
                ll = np.sum(model.log_likelihood(data, theta))
                print(f"Cycle {cycle + 1}/{self.n_cycles}: complete-data LL = {ll:.4f}")

        moments = None
        kept = n_iterations - max(self.burnin, first_cycle)
        if track and kept > 1:
            means = theta_sum / kept
            covariances = theta_outer / kept - means[:, :, None] * means[:, None, :]
            moments = (means, covariances * kept / (kept - 1))
        return n_iterations, converged, history, moments

    def _standard_errors(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        history: dict[str, list[NDArray[np.float64]]] | None,
        mean: NDArray[np.float64] | None = None,
        cov: NDArray[np.float64] | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], str | None, NDArray[np.float64] | None]:
        """Return standard errors, their method and the parameter covariance.

        Models with exact item derivatives use the observed information at
        the estimates, integrated over the ``N(mean, cov)`` ability density.
        Others fall back to the spread of the post-burn-in iterates in
        ``history``.
        """
        from mirt.estimation._louis_information import has_analytic_item_derivatives
        from mirt.estimation._patterns import (
            compress_responses,
            supports_pattern_compression,
        )
        from mirt.estimation.base import _parameter_bounds
        from mirt.estimation.quadrature import GaussHermiteQuadrature
        from mirt.estimation.standard_errors import estimate_covariance

        if not self.compute_standard_errors:
            return {}, None, None
        if has_analytic_item_derivatives(model):
            frequencies = None
            if supports_pattern_compression(model):
                responses, frequencies = compress_responses(responses)
            quadrature = GaussHermiteQuadrature(
                n_points=self.n_quadpts,
                n_dimensions=model.n_factors,
                mean=mean,
                cov=cov,
            )
            estimate = estimate_covariance(
                model,
                responses,
                quadrature,
                quadrature.weights,
                "oakes",
                frequencies=frequencies,
                bounds=lambda name: _parameter_bounds(model, name),
            )
            return estimate.standard_errors, "oakes", estimate.covariance
        errors = {
            name: np.std(chain, axis=0)
            for name, chain in (history or {}).items()
            if chain
        }
        return errors, "mhrm_iterate_sd" if errors else None, None

    def _sample_theta(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        rng: np.random.Generator,
        mean: NDArray[np.float64] | None = None,
        precision: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Metropolis-Hastings step for theta sampling.

        The ability density is standard normal unless ``precision``, the
        inverse covariance, and ``mean`` describe another normal density.
        """
        n_persons = theta.shape[0]

        proposal = theta + rng.normal(0, self.proposal_sd, theta.shape)

        ll_current = model.log_likelihood(responses, theta)
        ll_proposal = model.log_likelihood(responses, proposal)

        if precision is None:
            prior_current = stats.norm.logpdf(theta).sum(axis=1)
            prior_proposal = stats.norm.logpdf(proposal).sum(axis=1)
        else:
            current, proposed = theta - mean, proposal - mean
            prior_current = -0.5 * np.einsum("ni,ij,nj->n", current, precision, current)
            prior_proposal = -0.5 * np.einsum(
                "ni,ij,nj->n", proposed, precision, proposed
            )

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
        weights: NDArray[np.float64] | None = None,
    ) -> None:
        """Take one preconditioned Robbins-Monro step on every item.

        ``information`` holds each item's running complete-data information
        and is updated in place. Rows of ``responses`` and ``theta`` may stack
        several imputations, weighted by ``weights``. Fixed coordinates never
        enter the step. A step that would lower the item's complete-data
        likelihood on the current draw is halved until it does not, which
        keeps unit-gain burn-in steps from overshooting into degenerate
        regions; graded thresholds must also stay ordered.
        """
        updates: dict[int, NDArray[np.float64]] = {}
        for item in range(model.n_items):
            observed = responses[:, item] >= 0
            if not observed.any():
                continue
            params, bounds = self._get_item_params_and_bounds(model, item)
            if params.size == 0:
                continue
            objective = self._item_objective(
                model,
                item,
                theta[observed],
                responses[observed, item],
                params,
                bounds,
                None if weights is None else weights[observed],
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
            ordering = graded_threshold_constraint(model, item, params.size)
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
        weights: NDArray[np.float64] | None = None,
    ) -> _Objective:
        """Return an item's complete-data loss and gradient on the draws.

        Built-in families use their analytic kernels. Custom models evaluate
        their public probability hook and differentiate it numerically.
        Rows count with ``weights``, one each by default.
        """
        from mirt.estimation._mc_objective import _item_kernel

        if weights is None:
            weights = np.ones(len(responses))
        # Kernels may skip overflow guards inside their box, so the box must
        # cover the current values and the differencing steps around them.
        margin = 4 * _MHRM_DIFFERENCE_STEP * np.maximum(1.0, np.abs(params))
        box = [
            (min(low, value - pad), max(high, value + pad))
            for (low, high), value, pad in zip(bounds, params, margin, strict=True)
        ]
        kernel = _item_kernel(model, item, theta, responses, weights, box)
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
            return -float(np.sum(weights * np.log(np.clip(chosen, PROB_EPSILON, 1.0))))

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


def _largest_change(
    previous: Mapping[str, NDArray[np.float64]],
    current: Mapping[str, NDArray[np.float64]],
) -> float:
    """Return the largest absolute change between two parameter states.

    Padding stored as ``NaN`` in both states counts as unchanged.
    """
    change = 0.0
    for name, values in current.items():
        difference = np.abs(np.asarray(values, np.float64) - previous[name])
        change = max(change, float(np.max(np.nan_to_num(difference), initial=0.0)))
    return change


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
        if not model._item_indexed(name):
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


# ---------------------------------------------------------------------------
# Gibbs sampling
# ---------------------------------------------------------------------------


def _default_item_prior(name: str, lower: float) -> Prior | None:
    """Return the default Gibbs prior of an item parameter.

    ``None`` leaves the parameter uniform on its estimation box.
    """
    from mirt.estimation.priors import BetaPrior, LogNormalPrior, NormalPrior

    if name == "difficulty":
        return NormalPrior(0.0, 1.0)
    if name == "discrimination" and lower >= 0.0:
        # Density proportional to exp(-log(a)^2 / 2).
        return LogNormalPrior(1.0, 1.0)
    if name == "guessing":
        return BetaPrior(2.0, 8.0)
    if name == "upper":
        return BetaPrior(8.0, 2.0)
    return None


def _sampling_box(model: BaseItemModel, name: str) -> tuple[float, float]:
    """Return the box to which the Gibbs posterior of a parameter is restricted.

    Unidimensional 1PL-4PL models restrict slopes to ``[0.1, 5]`` and
    difficulties to ``[-6, 6]``, narrower than EM's boxes. Other parameters
    use their EM box.
    """
    from mirt.estimation.base import (
        _ITEM_PARAMETER_BOUNDS,
        _logistic_bounds,
        _parameter_bounds,
    )

    if (
        model.n_factors == 1
        and name in _ITEM_PARAMETER_BOUNDS
        and _logistic_bounds(model, name) is not None
    ):
        return _ITEM_PARAMETER_BOUNDS[name]
    return _parameter_bounds(model, name)


def _coordinate_kind(lower: float, upper: float) -> int:
    """Return the sampling scale of a coordinate with box ``[lower, upper]``."""
    if np.isfinite(lower) and np.isfinite(upper) and upper - lower <= 1.0:
        return _LOGIT
    if np.isfinite(lower) and lower >= 0.0:
        return _LOWER_LOG
    return _IDENTITY


class _ItemBlocks:
    """Free item coordinates of a model on an unconstrained sampling scale.

    Every free coordinate of every item has one entry of a flat vector,
    item after item in the layout of
    ``BaseEstimator._get_item_params_and_bounds``. A coordinate in a box at
    most one unit wide, such as a guessing parameter, is sampled as the
    logit of its position in the box; one with a nonnegative lower bound,
    such as a slope, as the log of its distance to it; others as they are.
    A graded item whose thresholds are all free samples its first threshold
    and the logs of the gaps between consecutive ones, which keeps them in
    order. Values outside the box, and graded thresholds out of order with
    fixed ones, have zero density.
    """

    def __init__(self, model: BaseItemModel, priors: Mapping[str, Prior]) -> None:
        masks = model.free_parameter_masks
        parameters = model.parameters
        names = [name for name in model._parameters if model._item_indexed(name)]
        canonical = {
            name: model._canonical_parameter_values(name, parameters[name]).reshape(
                model.n_items, -1
            )
            for name in names
        }
        item_of: list[int] = []
        lower: list[float] = []
        upper: list[float] = []
        kinds: list[int] = []
        previous: list[int] = []
        start: list[float] = []
        positions: dict[str, tuple[list[int], list[int]]] = {}
        self.item_slices: list[slice] = []
        self.orders: list[tuple[int, slice, Any]] = []
        depth: dict[int, int] = {}
        for item in range(model.n_items):
            first = len(item_of)
            for name in names:
                free = np.flatnonzero(
                    np.asarray(masks[name][item], dtype=np.bool_).reshape(-1)
                )
                if not free.size:
                    continue
                low, high = _sampling_box(model, name)
                coordinates, flat = positions.setdefault(name, ([], []))
                for column in free:
                    coordinates.append(len(item_of))
                    flat.append(item * canonical[name].shape[1] + int(column))
                    item_of.append(item)
                    lower.append(float(low))
                    upper.append(float(high))
                    kinds.append(_coordinate_kind(low, high))
                    previous.append(-1)
                    start.append(float(canonical[name][item, column]))
            last = len(item_of)
            self.item_slices.append(slice(first, last))
            order = graded_order(model, item, last - first)
            if order is None:
                continue
            if np.all(order.positions >= 0):
                index = first + order.positions
                kinds[index[0]] = _IDENTITY
                for k in range(1, len(index)):
                    kinds[index[k]] = _GAP
                    previous[index[k]] = int(index[k - 1])
                    depth[int(index[k])] = k
            else:
                self.orders.append((item, slice(first, last), order))

        self.n_items = model.n_items
        self.size = len(item_of)
        self.item_of = np.asarray(item_of, dtype=np.intp)
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.kinds = np.asarray(kinds, dtype=np.intp)
        self.previous = np.asarray(previous, dtype=np.intp)
        self.positions = {
            name: (np.asarray(coordinates, dtype=np.intp), np.asarray(flat, np.intp))
            for name, (coordinates, flat) in positions.items()
        }
        self.priors = [
            (coordinates, priors[name])
            for name, (coordinates, _) in self.positions.items()
            if name in priors
        ]
        self._log_scale = np.flatnonzero(self.kinds == _LOWER_LOG)
        self._logit = np.flatnonzero(self.kinds == _LOGIT)
        self._gaps = np.flatnonzero(self.kinds == _GAP)
        # Gaps are accumulated from the first threshold outward.
        self._levels = [
            np.asarray([index for index, k in depth.items() if k == level], np.intp)
            for level in range(1, max(depth.values(), default=0) + 1)
        ]
        self.start = self._inside(np.asarray(start, dtype=np.float64))

    def _inside(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Move values just inside their boxes and order graded thresholds."""
        width = self.upper - self.lower
        margin = 1e-3 * np.where(np.isfinite(width), np.minimum(width, 1.0), 1.0)
        values = np.where(
            np.isfinite(self.lower), np.maximum(values, self.lower + margin), values
        )
        values = np.where(
            np.isfinite(self.upper), np.minimum(values, self.upper - margin), values
        )
        for level in self._levels:
            values[level] = np.maximum(
                values[level], values[self.previous[level]] + 1e-3
            )
        return values

    def values(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map sampling-scale coordinates to parameter values."""
        values = np.array(unconstrained, dtype=np.float64)
        index = self._log_scale
        values[index] = self.lower[index] + np.exp(unconstrained[index])
        index = self._logit
        values[index] = self.lower[index] + (
            self.upper[index] - self.lower[index]
        ) * special.expit(unconstrained[index])
        for level in self._levels:
            values[level] = values[self.previous[level]] + np.exp(unconstrained[level])
        return values

    def unconstrained(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map parameter values inside their support to the sampling scale."""
        result = np.array(values, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            index = self._log_scale
            result[index] = np.log(values[index] - self.lower[index])
            index = self._logit
            result[index] = special.logit(
                (values[index] - self.lower[index])
                / (self.upper[index] - self.lower[index])
            )
            index = self._gaps
            result[index] = np.log(values[index] - values[self.previous[index]])
        return result

    def log_jacobian(self, unconstrained: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return each item's log-Jacobian of the map to parameter values."""
        terms = np.zeros(self.size)
        terms[self._log_scale] = unconstrained[self._log_scale]
        terms[self._gaps] = unconstrained[self._gaps]
        index = self._logit
        terms[index] = (
            np.log(self.upper[index] - self.lower[index])
            + special.log_expit(unconstrained[index])
            + special.log_expit(-unconstrained[index])
        )
        return np.bincount(self.item_of, terms, minlength=self.n_items)

    def log_prior(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return each item's log prior density."""
        total = np.zeros(self.n_items)
        for coordinates, prior in self.priors:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                density = np.asarray(prior.log_pdf(values[coordinates]), np.float64)
            total += np.bincount(
                self.item_of[coordinates], density, minlength=self.n_items
            )
        return total

    def outside(self, values: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Return the items whose values have zero density."""
        invalid = ~np.isfinite(values) | (values < self.lower) | (values > self.upper)
        gaps = self._gaps
        invalid[gaps] |= values[gaps] - values[self.previous[gaps]] < THRESHOLD_GAP
        items = np.bincount(self.item_of, invalid, minlength=self.n_items) > 0
        for item, block, order in self.orders:
            items[item] |= not order.satisfied(values[block])
        return items

    def write(self, model: BaseItemModel, values: NDArray[np.float64]) -> None:
        """Store coordinate values in ``model`` with one validated update."""
        current = model.parameters
        changed: dict[str, NDArray[np.float64]] = {}
        for name, (coordinates, flat) in self.positions.items():
            array = model._canonical_parameter_values(name, current[name])
            buffer = array.reshape(-1).copy()
            buffer[flat] = values[coordinates]
            changed[name] = model._canonical_parameter_values(
                name, buffer.reshape(array.shape)
            )
        if changed:
            model.set_parameters(**changed)


def _adaptation_windows(burnin: int) -> tuple[int, list[int]]:
    """Return when burn-in proposal covariance estimation starts and refits.

    As in Stan's warm-up, a first stretch adapts only the proposal scales,
    windows of doubling length then estimate the proposal covariance, and a
    final stretch adapts the scales to that covariance. Short burn-ins adapt
    scales only.
    """
    if burnin < 50:
        return burnin, []
    first = max(10, burnin * 15 // 100)
    last = burnin - max(10, burnin // 10)
    ends: list[int] = []
    position, length = first, 25
    while position + length <= last:
        end = position + length
        if last - end < 2 * length:
            end = last
        ends.append(end)
        position, length = end, 2 * length
    return first, ends


class _BlockProposals:
    """Random-walk proposals for every item block, adapted during burn-in.

    Item ``j`` proposes ``u + s_j L_j z`` with ``z`` standard normal. During
    burn-in ``log s_j`` follows a Robbins-Monro recursion toward an
    acceptance rate of 0.44 for one coordinate and 0.3 for more, and the
    ``L_j`` become Cholesky factors of the covariance of the draws in
    successive windows. Both are frozen after burn-in, so the kept draws
    come from a fixed Metropolis kernel.
    """

    def __init__(self, blocks: _ItemBlocks, burnin: int) -> None:
        sizes = np.array([block.stop - block.start for block in blocks.item_slices])
        self.groups: list[dict[str, Any]] = []
        for size in np.unique(sizes[sizes > 0]):
            items = np.flatnonzero(sizes == size)
            coordinates = np.stack(
                [
                    np.arange(blocks.item_slices[j].start, blocks.item_slices[j].stop)
                    for j in items
                ]
            )
            steps = np.where(
                blocks.kinds[coordinates] == _LOGIT, _GIBBS_LOGIT_STEP, _GIBBS_ITEM_STEP
            )
            self.groups.append(
                {
                    "items": items,
                    "coordinates": coordinates,
                    "factor": steps[:, :, None] * np.eye(int(size)),
                    "mean": np.zeros(coordinates.shape),
                    "m2": np.zeros((*coordinates.shape, int(size))),
                }
            )
        self.sizes = sizes
        self.target = np.where(sizes <= 1, 0.44, 0.3)
        self.log_scale = np.zeros(blocks.n_items)
        self.window_start, self.window_ends = _adaptation_windows(burnin)
        self.count = 0
        self.steps = 0

    def propose(
        self, unconstrained: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Return a proposal for every item."""
        proposal = unconstrained.copy()
        scale = np.exp(self.log_scale)
        for group in self.groups:
            noise = rng.standard_normal(group["coordinates"].shape)
            proposal[group["coordinates"]] += scale[group["items"], None] * np.einsum(
                "gij,gj->gi", group["factor"], noise
            )
        return proposal

    def adapt(
        self,
        iteration: int,
        log_ratio: NDArray[np.float64],
        unconstrained: NDArray[np.float64],
    ) -> None:
        """Adapt the proposals after burn-in iteration ``iteration``."""
        acceptance = np.exp(np.minimum(np.nan_to_num(log_ratio, nan=-np.inf), 0.0))
        self.steps += 1
        self.log_scale += self.steps**-_ADAPTATION_DECAY * (acceptance - self.target)
        if not self.window_ends or iteration < self.window_start:
            return
        self.count += 1
        for group in self.groups:
            draws = unconstrained[group["coordinates"]]
            delta = draws - group["mean"]
            group["mean"] += delta / self.count
            group["m2"] += delta[:, :, None] * (draws - group["mean"])[:, None, :]
        if iteration + 1 not in self.window_ends:
            return
        for group in self.groups:
            size = group["coordinates"].shape[1]
            covariance = group["m2"] / max(self.count - 1, 1)
            covariance = (self.count / (self.count + 5.0)) * covariance + 1e-3 * (
                5.0 / (self.count + 5.0)
            ) * np.eye(size)
            group["factor"] = _cholesky_factors(covariance, group["factor"])
            group["mean"][:] = 0.0
            group["m2"][:] = 0.0
            self.log_scale[group["items"]] = np.log(2.38 / np.sqrt(size))
        self.count = 0
        self.steps = 0


def _cholesky_factors(
    covariance: NDArray[np.float64], fallback: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return Cholesky factors of stacked covariances, keeping ``fallback``.

    Items whose covariance is not positive definite keep their factor.
    """
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        factors = fallback.copy()
        for index, matrix in enumerate(covariance):
            try:
                factors[index] = np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError:
                continue
        return factors


def _starting_abilities(
    model: BaseItemModel, responses: NDArray[np.int_]
) -> NDArray[np.float64]:
    """Return starting abilities: normal scores of the observed mean scores.

    Ranking persons by their mean item score, each item scaled to ``[0, 1]``,
    places the chain near the ability posterior of a unidimensional model,
    which shortens the slow joint drift of abilities and item locations from
    a degenerate start. Multidimensional models start at zero.
    """
    n_persons = responses.shape[0]
    if model.n_factors != 1 or n_persons < 2:
        return np.zeros((n_persons, model.n_factors))
    observed = responses >= 0
    top = np.maximum(np.max(np.where(observed, responses, 0), axis=0), 1)
    scaled = np.where(observed, responses / top, 0.0)
    counts = observed.sum(axis=1)
    scores = np.divide(
        scaled.sum(axis=1),
        counts,
        out=np.full(n_persons, 0.5),
        where=counts > 0,
    )
    ranks = stats.rankdata(scores)
    return special.ndtri((ranks - 0.5) / n_persons)[:, None]


class _ItemLikelihoods:
    """Person-by-item log-likelihoods of a response matrix.

    Built-in likelihoods evaluate the public probability curves of all
    items at once. Models with their own ``log_likelihood`` are evaluated
    one item at a time, with the other items' responses marked missing.
    """

    def __init__(self, model: BaseItemModel, responses: NDArray[np.int_]) -> None:
        from mirt.estimation._mc_likelihood import uses_default_sample_likelihood

        self.responses = responses
        self.vectorized = uses_default_sample_likelihood(model)
        self.polytomous = bool(model.is_polytomous)
        if self.vectorized and not self.polytomous:
            self.success = (responses == 1).astype(np.float64)
            self.failure = (responses == 0).astype(np.float64)
        elif self.vectorized:
            self.observed = [
                np.flatnonzero(responses[:, item] >= 0) for item in range(model.n_items)
            ]

    def matrix(
        self, model: BaseItemModel, theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the ``(n_persons, n_items)`` log-likelihood contributions."""
        n_persons, n_items = self.responses.shape
        if not self.vectorized:
            columns = np.empty((n_persons, n_items))
            for item in range(n_items):
                masked = np.full_like(self.responses, -1)
                masked[:, item] = self.responses[:, item]
                columns[:, item] = model.log_likelihood(masked, theta)
            return columns
        if not self.polytomous:
            probability = np.broadcast_to(
                np.asarray(model.probability(theta), dtype=np.float64),
                (n_persons, n_items),
            )
            probability = np.clip(probability, PROB_EPSILON, 1.0 - PROB_EPSILON)
            return self.success * np.log(probability) + self.failure * np.log1p(
                -probability
            )
        columns = np.zeros((n_persons, n_items))
        for item, rows in enumerate(self.observed):
            if not rows.size:
                continue
            probability = np.asarray(model.probability(theta, item), np.float64)
            chosen = probability[rows, self.responses[rows, item]]
            columns[rows, item] = np.log(np.clip(chosen, PROB_EPSILON, 1.0))
        return columns


class GibbsSampler(BaseEstimator):
    """Full Bayesian estimation via Metropolis-within-Gibbs sampling.

    Every iteration updates

    1. each person's abilities, by a random-walk Metropolis step against
       their standard normal prior, and
    2. each item's free parameters, by a random-walk Metropolis step on an
       unconstrained scale: the log of a slope's distance to its lower
       bound, the logit of a guessing or upper asymptote within its box,
       and the first graded threshold with the logs of the gaps between
       consecutive ones. Given the abilities the items are independent, so
       every item is accepted or rejected on its own likelihood, and the
       Jacobian of the scale enters the acceptance ratio.

    Proposal scales, and the proposal covariance of each item, adapt during
    burn-in and stay fixed afterwards. Coordinates fixed with
    ``set_free_parameter_masks`` keep their values.

    Item priors default to ``N(0, 1)`` for difficulty, ``LogNormal(1, 1)``
    (density proportional to ``exp(-log(a)**2 / 2)``) for positive
    discrimination, ``Beta(2, 8)`` for guessing and ``Beta(8, 2)`` for upper
    asymptotes; other item parameters are uniform on the estimation box of
    their family, which for the unidimensional 1PL-4PL is ``[0.1, 5]`` for
    discrimination and ``[-6, 6]`` for difficulty. Chains start from the
    model's values and, for one factor, from abilities ranked by the
    persons' mean item scores. Every family, the 2PL included, is sampled in
    NumPy.

    With ``n_chains > 1`` the draws of several seeded chains are stacked;
    chains then start from those values jittered on the sampling scale, so R-hat can detect chains that still remember their starts.
    Abilities and item parameters drift slowly as a whole, so chains of a
    few thousand iterations are typical. ``parallel_chains=True`` runs NumPy
    chains in worker processes. ``MCMCResult.converged`` applies the R-hat
    and ESS thresholds of Vehtari et al. (2021) within those chains.

    ``MCMCResult.log_likelihood`` is the marginal log-likelihood at the
    posterior-mean item parameters: for up to three factors it is integrated
    on EM's default grid, so it equals EM's log-likelihood at the same
    parameters; beyond that it is estimated by importance sampling from
    normal approximations of each person's ability posterior. DIC and WAIC
    are Bayesian criteria built from the conditional likelihood of the draws.
    """

    def __init__(
        self,
        n_iter: int = 5000,
        burnin: int = 1000,
        thin: int = 1,
        n_chains: int = 1,
        priors: PriorSpecification | Mapping[str, Prior] | None = None,
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
        priors : PriorSpecification or mapping of str to Prior, optional
            Item-parameter priors that replace the defaults (see the class
            notes), by stored parameter name, for example
            ``{"guessing": BetaPrior(5, 17)}``. A ``PriorSpecification`` sets
            its discrimination, difficulty, guessing and upper priors, its
            omitted discrimination and difficulty fields meaning
            ``LogNormal(0, 0.5)`` and ``N(0, 2)``; its ``theta`` prior must
            stay standard normal.
        verbose : bool
            Whether to print progress
        use_rust : bool
            Accepted for backward compatibility and ignored: the sampler
            always runs in NumPy
        seed : int, optional
            Random seed for reproducibility
        parallel_chains : bool
            Whether to run multiple chains in spawned worker processes; the
            draws equal those of a serial run

        Raises
        ------
        MirtValidationError
            If a control is invalid or ``priors`` is not a
            ``PriorSpecification`` or a mapping of names to ``Prior`` objects.
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
        self.priors = _validate_gibbs_priors(priors)
        self.use_rust = use_rust
        self.seed = seed
        self.parallel_chains = parallel_chains

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> MCMCResult:
        """Fit model using Gibbs sampling.

        Parameters
        ----------
        model : BaseItemModel
            IRT model. Coordinates fixed with ``set_free_parameter_masks``
            keep their values.
        responses : NDArray
            Response matrix

        Returns
        -------
        MCMCResult
            MCMC estimation result with chains and diagnostics

        Raises
        ------
        MirtModelError
            If the model has free parameters shared by all items, such as
            rating-scale thresholds, which the itemwise updates cannot move.
        MirtValidationError
            If ``priors`` name parameters the model does not have.
        """
        responses = self._validate_responses(responses, model.n_items, model)
        n_persons = responses.shape[0]
        if not model._parameters:
            model._initialize_parameters()

        self._check_shared_parameters(model)
        self._item_priors(model, check=True)
        if self.n_chains > 1:
            chain_arrays = self._run_chains(model, responses, n_persons)
        else:
            chain_arrays = self._run_single_chain(
                model, responses, n_persons, self.seed
            )

        for name in model.parameters:
            model._parameters[name] = np.mean(chain_arrays[name], axis=0)
        return self._result(model, responses, chain_arrays)

    def _result(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        chains: dict[str, NDArray[np.float64]],
    ) -> MCMCResult:
        """Summarize posterior draws for a model set to their means."""
        model._is_fitted = True
        _, rhat, ess, ess_tail = _diagnose_chains(chains, model, self.n_chains)
        posterior = (
            _theta_moments(chains["theta"])
            if model.n_factors > _MAX_QUADRATURE_FACTORS
            else None
        )
        log_likelihood = _marginal_log_likelihood(
            model,
            responses,
            posterior=posterior,
            rng=np.random.default_rng(self.seed),
        )
        return MCMCResult(
            model=model,
            chains=chains,
            log_likelihood=log_likelihood,
            dic=self._compute_dic(chains, model, responses),
            waic=self._compute_waic(chains, model, responses),
            rhat=rhat,
            ess=ess,
            n_iterations=self.n_iter,
            burnin=self.burnin,
            thin=self.thin,
            n_chains=self.n_chains,
            ess_tail=ess_tail,
        )

    def _item_priors(
        self, model: BaseItemModel, check: bool = False
    ) -> dict[str, Prior]:
        """Return the prior of every item parameter that has one.

        ``check`` warns about, or rejects, a ``PriorSpecification`` that
        does not fit the model.
        """
        from mirt.estimation._item_priors import resolve_item_priors

        priors: dict[str, Prior] = {}
        for name in model._parameters:
            if not model._item_indexed(name):
                continue
            default = _default_item_prior(name, _sampling_box(model, name)[0])
            if default is not None:
                priors[name] = default
        priors.update(
            resolve_item_priors(self.priors or None, model, check_specification=check)
        )
        return priors

    def _run_single_chain(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        n_persons: int,
        seed: int | None,
        disperse: bool = False,
    ) -> dict[str, NDArray]:
        """Run a single MCMC chain.

        The chain starts from the model's values and the abilities of
        :func:`_starting_abilities`; ``disperse`` jitters both on the
        sampling scale.
        """
        rng = np.random.default_rng(seed)
        blocks = _ItemBlocks(model, self._item_priors(model))
        proposals = _BlockProposals(blocks, self.burnin)
        likelihoods = _ItemLikelihoods(model, responses)
        n_items, n_factors = model.n_items, model.n_factors

        unconstrained = blocks.unconstrained(blocks.start)
        values = blocks.start
        theta = _starting_abilities(model, responses)
        if disperse:
            jittered = unconstrained + _DISPERSION_SD * rng.standard_normal(blocks.size)
            moved = blocks.values(jittered)
            keep = ~blocks.outside(moved)[blocks.item_of]
            unconstrained = np.where(keep, jittered, unconstrained)
            values = np.where(keep, moved, values)
            theta = theta + _DISPERSION_SD * rng.standard_normal(theta.shape)
        blocks.write(model, values)
        proposal_model = model.copy()
        log_prior = blocks.log_prior(values)
        log_jacobian = blocks.log_jacobian(unconstrained)
        contributions = likelihoods.matrix(model, theta)
        theta_scale = np.full(n_persons, _GIBBS_THETA_STEP)
        theta_target = 0.44 if n_factors == 1 else 0.3

        chains: dict[str, list] = {name: [] for name in model.parameters}
        chains["theta"] = []
        chains["log_likelihood"] = []

        for iteration in range(self.n_iter):
            adapting = iteration < self.burnin
            # Abilities given the items.
            proposal = theta + theta_scale[:, None] * rng.standard_normal(theta.shape)
            proposed = likelihoods.matrix(model, proposal)
            theta_ratio = (
                proposed.sum(axis=1)
                - contributions.sum(axis=1)
                - 0.5 * (np.sum(proposal**2, axis=1) - np.sum(theta**2, axis=1))
            )
            accepted = np.log(rng.random(n_persons)) < theta_ratio
            theta = np.where(accepted[:, None], proposal, theta)
            contributions[accepted] = proposed[accepted]

            # Items given the abilities, each accepted on its own.
            candidate = proposals.propose(unconstrained, rng)
            candidate_values = blocks.values(candidate)
            invalid = blocks.outside(candidate_values)
            reset = invalid[blocks.item_of]
            candidate_values[reset] = values[reset]
            blocks.write(proposal_model, candidate_values)
            proposed = likelihoods.matrix(proposal_model, theta)
            candidate_prior = blocks.log_prior(candidate_values)
            candidate_jacobian = blocks.log_jacobian(candidate)
            with np.errstate(invalid="ignore"):
                item_ratio = (
                    proposed.sum(axis=0)
                    - contributions.sum(axis=0)
                    + candidate_prior
                    - log_prior
                    + candidate_jacobian
                    - log_jacobian
                )
            item_ratio[invalid] = -np.inf
            accept = np.log(rng.random(n_items)) < item_ratio
            if accept.any():
                take = accept[blocks.item_of]
                unconstrained = np.where(take, candidate, unconstrained)
                values = np.where(take, candidate_values, values)
                log_prior = np.where(accept, candidate_prior, log_prior)
                log_jacobian = np.where(accept, candidate_jacobian, log_jacobian)
                contributions[:, accept] = proposed[:, accept]
                blocks.write(model, values)

            if adapting:
                rate = (iteration + 1) ** -_ADAPTATION_DECAY
                acceptance = np.exp(
                    np.minimum(np.nan_to_num(theta_ratio, nan=-np.inf), 0.0)
                )
                theta_scale *= np.exp(rate * (acceptance - theta_target))
                proposals.adapt(iteration, item_ratio, unconstrained)

            if iteration >= self.burnin and (iteration - self.burnin) % self.thin == 0:
                for name, parameter in model.parameters.items():
                    chains[name].append(parameter)
                chains["theta"].append(theta.copy())
                chains["log_likelihood"].append(float(contributions.sum()))

            if self.verbose and (iteration + 1) % 500 == 0:
                ll = float(contributions.sum())
                print(f"Iteration {iteration + 1}/{self.n_iter}: LL = {ll:.4f}")

        return {name: np.array(chain) for name, chain in chains.items()}

    def _run_chains(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        n_persons: int,
    ) -> dict[str, NDArray]:
        """Run ``n_chains`` seeded chains from dispersed starts and stack draws.

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
                        self._run_single_chain,
                        model.copy(),
                        responses,
                        n_persons,
                        seed,
                        True,
                    )
                    for seed in seeds
                ]
                all_chains = [future.result() for future in futures]
        else:
            all_chains = [
                self._run_single_chain(model.copy(), responses, n_persons, seed, True)
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


def _validate_gibbs_priors(
    priors: Any,
) -> PriorSpecification | dict[str, Prior]:
    """Return validated Gibbs item priors; ``{}`` keeps the defaults."""
    from mirt.estimation._item_priors import validate_item_priors
    from mirt.estimation.priors import NormalPrior, PriorSpecification

    try:
        validated = validate_item_priors(priors)
    except MirtValidationError as exc:
        raise MirtValidationError(
            "priors must be a PriorSpecification or a mapping of parameter "
            "names to Prior objects, such as {'difficulty': NormalPrior(0, 2)}",
            parameter="priors",
            value=type(priors).__name__,
            expected="PriorSpecification or Mapping[str, Prior]",
        ) from exc
    if isinstance(validated, PriorSpecification):
        theta = validated.theta
        if not (
            type(theta) is NormalPrior
            and float(theta.mu) == 0.0
            and float(theta.sigma) == 1.0
        ):
            raise MirtValidationError(
                "GibbsSampler keeps the ability prior standard normal; leave "
                "PriorSpecification.theta at NormalPrior(0, 1)",
                parameter="priors",
                value=repr(theta),
                expected="NormalPrior(mu=0, sigma=1)",
            )
    return {} if validated is None else validated
