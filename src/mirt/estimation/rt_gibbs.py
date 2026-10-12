from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logsumexp

from mirt.backends.rust.response_time import (
    rt_accept_person_proposals,
    rt_joint_log_likelihood,
)
from mirt.constants import PROB_EPSILON
from mirt.estimation._identified_covariance import sample_unit_leading_covariance
from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.models.response_time import ResponseTimeModel, ResponseTimeResult

# Random-walk steps per sweep for log time discrimination. Each step only
# touches per-item sufficient statistics, and five steps recover most of the
# mixing of a direct draw.
_TIME_DISCRIMINATION_MH_STEPS = 5
# Random-walk scales for the accuracy item updates: (log a, b) jointly, and
# guessing on the logit scale of c / guess_upper.
_ACCURACY_PROPOSAL_SD = 0.1
_GUESSING_LOGIT_PROPOSAL_SD = 0.3
_GUESSING_LOGIT_LIMIT = 30.0
# Person-by-item cells evaluated at once by the vectorized item updates.
_ITEM_UPDATE_MAX_VALUES = 1_000_000
# Population parameters fixed by the identification constraints. They stay in
# ``chains`` as constants but are left out of the convergence diagnostics.
_FIXED_POPULATION_PARAMETERS = ("mu_theta", "mu_tau", "sigma_11")


def _accuracy_log_likelihoods(
    correct: NDArray[np.bool_],
    observed: NDArray[np.bool_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing_sets: Sequence[NDArray[np.float64] | None],
) -> NDArray[np.float64]:
    """Sum every item's accuracy log likelihood over persons.

    The logistic curve is evaluated once per person chunk and shared by all
    guessing vectors in ``guessing_sets``; ``None`` selects the 2PL. Each
    cell takes the log of its clipped response probability, as
    :func:`rt_joint_log_likelihood` does, so every Gibbs block targets the
    same likelihood.

    Returns
    -------
    NDArray
        Shape ``(len(guessing_sets), n_items)``.
    """
    n_persons, n_items = correct.shape
    totals = np.zeros((len(guessing_sets), n_items), dtype=np.float64)
    rows_per_chunk = max(1, _ITEM_UPDATE_MAX_VALUES // max(n_items, 1))
    for start in range(0, n_persons, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_persons)
        chunk_correct = correct[start:stop]
        chunk_observed = observed[start:stop]
        logistic = expit(discrimination * (theta[start:stop, None] - difficulty))
        complement = 1.0 - logistic
        for set_idx, guessing in enumerate(guessing_sets):
            if guessing is None:
                probability = np.where(chunk_correct, logistic, complement)
            else:
                probability = np.where(
                    chunk_correct,
                    guessing + (1.0 - guessing) * logistic,
                    (1.0 - guessing) * complement,
                )
            np.clip(probability, PROB_EPSILON, 1.0 - PROB_EPSILON, out=probability)
            np.log(probability, out=probability)
            probability *= chunk_observed
            totals[set_idx] += np.sum(probability, axis=0)
    return totals


@dataclass
class RTModelPriors:
    """Prior distributions for response time model parameters.

    Item parameters have independent normal priors: ``log a_j`` and ``b_j``
    for accuracy, ``log alpha_j`` (time discrimination) and ``beta_j`` (time
    intensity) for speed. A 3PL guessing parameter ``c_j`` has a
    ``Beta(guess_alpha, guess_beta)`` prior truncated to
    ``(0, guess_upper)``. The person ability-speed pairs are
    ``N(0, Sigma)`` with ``Var(theta) = Sigma[0, 0] = 1``, which identifies
    the model. ``Sigma`` has the inverse-Wishart prior
    ``IW(sigma_df, sigma_scale)`` conditioned on that unit ability variance.

    Attributes
    ----------
    disc_mean : float
        Prior mean for log discrimination
    disc_var : float
        Prior variance for log discrimination
    diff_mean : float
        Prior mean for difficulty
    diff_var : float
        Prior variance for difficulty
    time_disc_mean : float
        Prior mean for log time discrimination
    time_disc_var : float
        Prior variance for log time discrimination
    time_int_mean : float
        Prior mean for time intensity
    time_int_var : float
        Prior variance for time intensity
    mu_mean : NDArray, optional
        Deprecated and ignored. The ability-speed mean is fixed at zero to
        identify the model. Passing a value other than the former default
        ``[0, 0]`` emits a ``DeprecationWarning``.
    mu_cov : NDArray, optional
        Deprecated and ignored, like ``mu_mean``. Passing a value other than
        the former default ``10 * I`` emits a ``DeprecationWarning``.
    sigma_df : int
        Degrees of freedom for inverse-Wishart prior on Σ. It only affects
        the ability-speed covariance, not the item parameters.
    sigma_scale : NDArray
        Scale matrix for inverse-Wishart prior on Σ
    guess_alpha : float
        First Beta shape parameter of the 3PL guessing prior
    guess_beta : float
        Second Beta shape parameter of the 3PL guessing prior
    guess_upper : float
        Upper bound of the guessing parameters, in ``(0, 1]``. The default
        ``0.5`` with ``guess_alpha = guess_beta = 1`` is a uniform prior on
        ``(0, 0.5)``.
    """

    disc_mean: float = 0.0
    disc_var: float = 1.0
    diff_mean: float = 0.0
    diff_var: float = 4.0
    time_disc_mean: float = 0.0
    time_disc_var: float = 1.0
    time_int_mean: float = 0.0
    time_int_var: float = 4.0
    mu_mean: NDArray[np.float64] | None = None
    mu_cov: NDArray[np.float64] | None = None
    sigma_df: int = 4
    sigma_scale: NDArray[np.float64] | None = None
    guess_alpha: float = 1.0
    guess_beta: float = 1.0
    guess_upper: float = 0.5

    def __post_init__(self) -> None:
        for name in ("disc_mean", "diff_mean", "time_disc_mean", "time_int_mean"):
            value = getattr(self, name)
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    f"{name} must be finite",
                    parameter=name,
                    value=value,
                    expected="finite number",
                ) from exc
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(numeric):
                raise MirtValidationError(
                    f"{name} must be finite",
                    parameter=name,
                    value=value,
                    expected="finite number",
                )
            setattr(self, name, numeric)

        for name in (
            "disc_var",
            "diff_var",
            "time_disc_var",
            "time_int_var",
            "guess_alpha",
            "guess_beta",
        ):
            value = getattr(self, name)
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    f"{name} must be finite and positive",
                    parameter=name,
                    value=value,
                    expected="> 0",
                ) from exc
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isfinite(numeric)
                or numeric <= 0.0
            ):
                raise MirtValidationError(
                    f"{name} must be finite and positive",
                    parameter=name,
                    value=value,
                    expected="> 0",
                )
            setattr(self, name, numeric)

        if (
            isinstance(self.sigma_df, (bool, np.bool_))
            or not isinstance(self.sigma_df, (int, np.integer))
            or self.sigma_df < 2
        ):
            raise MirtValidationError(
                "sigma_df must be an integer of at least 2",
                parameter="sigma_df",
                value=self.sigma_df,
                expected="integer >= 2",
            )
        self.sigma_df = int(self.sigma_df)

        try:
            guess_upper = float(self.guess_upper)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "guess_upper must be in (0, 1]",
                parameter="guess_upper",
                value=self.guess_upper,
                expected="0 < guess_upper <= 1",
            ) from exc
        if (
            isinstance(self.guess_upper, (bool, np.bool_))
            or not np.isfinite(guess_upper)
            or not 0.0 < guess_upper <= 1.0
        ):
            raise MirtValidationError(
                "guess_upper must be in (0, 1]",
                parameter="guess_upper",
                value=self.guess_upper,
                expected="0 < guess_upper <= 1",
            )
        self.guess_upper = guess_upper

        # The population mean is fixed at zero for identification, so these
        # former priors are validated for compatibility but otherwise unused.
        if self.mu_mean is None:
            self.mu_mean = np.zeros(2)
        else:
            self.mu_mean = self._validated_array("mu_mean", self.mu_mean, (2,))
            self._warn_ignored_mean_prior("mu_mean", self.mu_mean, np.zeros(2))
        if self.mu_cov is None:
            self.mu_cov = np.eye(2) * 10
        else:
            self.mu_cov = self._validated_array(
                "mu_cov", self.mu_cov, (2, 2), positive_definite=True
            )
            self._warn_ignored_mean_prior("mu_cov", self.mu_cov, np.eye(2) * 10)
        if self.sigma_scale is None:
            self.sigma_scale = np.eye(2)
        else:
            self.sigma_scale = self._validated_array(
                "sigma_scale", self.sigma_scale, (2, 2), positive_definite=True
            )

    @staticmethod
    def _warn_ignored_mean_prior(
        name: str,
        value: NDArray[np.float64],
        former_default: NDArray[np.float64],
    ) -> None:
        """Warn when a population-mean prior that no longer applies is set."""
        if not np.array_equal(value, former_default):
            warnings.warn(
                f"{name} is deprecated and ignored: the ability-speed mean is "
                "fixed at zero to identify the response-time model",
                DeprecationWarning,
                stacklevel=4,
            )

    @staticmethod
    def _validated_array(
        name: str,
        values: NDArray[np.float64],
        shape: tuple[int, ...],
        *,
        positive_definite: bool = False,
    ) -> NDArray[np.float64]:
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"{name} must be numeric",
                parameter=name,
                value=values,
                expected=f"finite array with shape {shape}",
            ) from exc
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise MirtValidationError(
                f"{name} must be a finite array with shape {shape}",
                parameter=name,
                value=array.shape,
                expected=f"finite array with shape {shape}",
            )
        if positive_definite:
            if not np.allclose(array, array.T, rtol=0.0, atol=1e-12):
                raise MirtValidationError(
                    f"{name} must be symmetric positive definite",
                    parameter=name,
                    value=array,
                    expected="symmetric positive-definite matrix",
                )
            try:
                np.linalg.cholesky(array)
            except np.linalg.LinAlgError as exc:
                raise MirtValidationError(
                    f"{name} must be symmetric positive definite",
                    parameter=name,
                    value=array,
                    expected="symmetric positive-definite matrix",
                ) from exc
        return array.copy()


class ResponseTimeGibbsSampler:
    """Gibbs sampler for joint response time models.

    Implements a blocked Gibbs sampler for the Van der Linden hierarchical
    model with MCMC estimation of all parameters.

    The accuracy likelihood is invariant to a common shift and rescaling of
    ability and difficulty, and the log-normal time model to a common shift
    of speed and time intensity. The sampler fixes those directions as in
    van der Linden (2007) and Fox, Klein Entink and van der Linden (2007):
    the ability and speed means are zero and the ability variance is one.
    Only ``Var(tau)`` and ``Cov(theta, tau)`` are sampled, so
    ``mu_theta``, ``mu_tau`` and ``sigma_11`` are constant in the chains and
    left out of ``rhat`` and ``ess``.

    Parameters
    ----------
    n_iter : int
        Total number of MCMC iterations
    burnin : int
        Number of burn-in iterations to discard
    thin : int
        Thinning interval
    n_chains : int
        Number of independent chains
    priors : RTModelPriors, optional
        Prior specifications
    proposal_sd : float
        Standard deviation for MH proposals
    adapt_interval : int
        Interval for adapting proposal SD
    verbose : bool
        Print progress
    seed : int, optional
        Random seed
    use_rust : bool, default=True
        Use compiled likelihood and person-proposal evaluation when available.
    """

    def __init__(
        self,
        n_iter: int = 5000,
        burnin: int = 1000,
        thin: int = 1,
        n_chains: int = 1,
        priors: RTModelPriors | None = None,
        proposal_sd: float = 0.3,
        adapt_interval: int = 100,
        verbose: bool = False,
        seed: int | None = None,
        use_rust: bool = True,
    ) -> None:
        for name, value in (
            ("n_iter", n_iter),
            ("thin", thin),
            ("n_chains", n_chains),
            ("adapt_interval", adapt_interval),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise MirtValidationError(
                    f"{name} must be a positive integer",
                    parameter=name,
                    value=value,
                    expected="positive integer",
                )
        if (
            isinstance(burnin, (bool, np.bool_))
            or not isinstance(burnin, (int, np.integer))
            or burnin < 0
        ):
            raise MirtValidationError(
                "burnin must be a non-negative integer",
                parameter="burnin",
                value=burnin,
                expected="non-negative integer",
            )
        if burnin >= n_iter:
            raise MirtValidationError(
                "burnin must be less than n_iter",
                parameter="burnin",
                value=burnin,
                expected=f"< {n_iter}",
            )
        if priors is not None and not isinstance(priors, RTModelPriors):
            raise MirtValidationError(
                "priors must be an RTModelPriors instance or None",
                parameter="priors",
                value=type(priors).__name__,
                expected="RTModelPriors or None",
            )
        try:
            proposal_sd_value = float(proposal_sd)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "proposal_sd must be finite and positive",
                parameter="proposal_sd",
                value=proposal_sd,
                expected="> 0",
            ) from exc
        if (
            isinstance(proposal_sd, (bool, np.bool_))
            or not np.isfinite(proposal_sd_value)
            or proposal_sd_value <= 0.0
        ):
            raise MirtValidationError(
                "proposal_sd must be finite and positive",
                parameter="proposal_sd",
                value=proposal_sd,
                expected="> 0",
            )
        if not isinstance(verbose, (bool, np.bool_)):
            raise MirtValidationError(
                "verbose must be a boolean",
                parameter="verbose",
                value=verbose,
                expected="boolean",
            )
        if seed is not None and (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise MirtValidationError(
                "seed must be a non-negative integer or None",
                parameter="seed",
                value=seed,
                expected="non-negative integer or None",
            )
        if not isinstance(use_rust, (bool, np.bool_)):
            raise MirtValidationError(
                "use_rust must be a boolean",
                parameter="use_rust",
                value=use_rust,
                expected="boolean",
            )

        self.n_iter = int(n_iter)
        self.burnin = int(burnin)
        self.thin = int(thin)
        self.n_chains = int(n_chains)
        self.priors = RTModelPriors() if priors is None else priors
        self.proposal_sd = proposal_sd_value
        self.adapt_interval = int(adapt_interval)
        self.verbose = bool(verbose)
        self.seed = None if seed is None else int(seed)
        self.use_rust = bool(use_rust)

    def fit(
        self,
        responses: NDArray[np.int_] | NDArray[np.float64],
        response_times: NDArray[np.float64],
        accuracy_model: Literal["2PL", "3PL"] = "2PL",
    ) -> ResponseTimeResult:
        """Fit response time model via Gibbs sampling.

        Parameters
        ----------
        responses : NDArray
            Binary response matrix (n_persons, n_items)
        response_times : NDArray
            Response time matrix (n_persons, n_items) in seconds
        accuracy_model : str
            IRT model for accuracy ("2PL" or "3PL")

        Returns
        -------
        ResponseTimeResult
            Fitted model with posterior samples. ``log_likelihood``, ``dic``
            and ``waic`` evaluate every retained draw with the item and person
            parameters of the same iteration.
        """
        from mirt.models.response_time import ResponseTimeModel, ResponseTimeResult

        responses, log_rt = self._validate_fit_data(
            responses, response_times, accuracy_model
        )
        n_persons, n_items = responses.shape
        chain_runs = [
            self._run_chain(
                responses,
                log_rt,
                accuracy_model,
                rng,
                chain_idx,
            )
            for chain_idx, rng in enumerate(self._chain_generators())
        ]
        diagnostic_chains = {
            name: np.stack([run[0][name] for run in chain_runs], axis=0)
            for name in chain_runs[0][0]
        }
        chains = {
            name: values.reshape((-1,) + values.shape[2:])
            for name, values in diagnostic_chains.items()
        }
        theta_by_chain = np.stack([run[1] for run in chain_runs], axis=0)
        tau_by_chain = np.stack([run[2] for run in chain_runs], axis=0)
        theta_samples = theta_by_chain.reshape(-1, n_persons)
        tau_samples = tau_by_chain.reshape(-1, n_persons)

        theta_est = np.mean(theta_samples, axis=0)
        tau_est = np.mean(tau_samples, axis=0)
        theta_se = np.std(theta_samples, axis=0)
        tau_se = np.std(tau_samples, axis=0)

        model = ResponseTimeModel(
            n_items=n_items,
            accuracy_model=accuracy_model,
            discrimination=np.mean(chains["discrimination"], axis=0),
            difficulty=np.mean(chains["difficulty"], axis=0),
            guessing=np.mean(chains["guessing"], axis=0)
            if accuracy_model == "3PL"
            else None,
            time_discrimination=np.mean(chains["time_discrimination"], axis=0),
            time_intensity=np.mean(chains["time_intensity"], axis=0),
            ability_speed_mean=np.zeros(2),
            ability_speed_cov=np.array(
                [
                    [1.0, np.mean(chains["sigma_12"])],
                    [np.mean(chains["sigma_12"]), np.mean(chains["sigma_22"])],
                ]
            ),
            use_rust=self.use_rust,
        )

        sample_log_likelihoods = self._draw_log_likelihoods(
            responses, log_rt, theta_samples, tau_samples, chains
        )
        log_likelihood = float(np.mean(np.sum(sample_log_likelihoods, axis=1)))

        dic = self._compute_dic(
            model,
            responses,
            log_rt,
            theta_samples,
            tau_samples,
            sample_log_likelihoods,
        )
        waic = self._compute_waic(
            model,
            responses,
            log_rt,
            theta_samples,
            tau_samples,
            sample_log_likelihoods,
        )

        sampled_chains = {
            name: values
            for name, values in diagnostic_chains.items()
            if name not in _FIXED_POPULATION_PARAMETERS
        }
        rhat = self._compute_rhat(sampled_chains)
        ess = self._compute_ess(sampled_chains)

        converged = all(np.isfinite(r) and r < 1.1 for r in rhat.values())

        return ResponseTimeResult(
            model=model,
            theta_estimates=theta_est,
            tau_estimates=tau_est,
            theta_se=theta_se,
            tau_se=tau_se,
            chains=chains,
            log_likelihood=log_likelihood,
            dic=dic,
            waic=waic,
            rhat=rhat,
            ess=ess,
            n_iterations=self.n_iter,
            n_chains=self.n_chains,
            converged=converged,
        )

    def _validate_fit_data(
        self,
        responses: NDArray[np.int_] | NDArray[np.float64],
        response_times: NDArray[np.float64],
        accuracy_model: str,
    ) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
        """Validate and normalize observed accuracy and timing data."""
        if not isinstance(accuracy_model, str) or accuracy_model not in ("2PL", "3PL"):
            raise MirtValidationError(
                "accuracy_model must be '2PL' or '3PL'",
                parameter="accuracy_model",
                value=accuracy_model,
                expected="'2PL' or '3PL'",
            )

        response_values = np.asarray(responses)
        if response_values.ndim != 2:
            raise MirtValidationError(
                "responses must have shape (n_persons, n_items)",
                parameter="responses",
                value=response_values.shape,
                expected="two-dimensional matrix",
            )
        if response_values.shape[0] == 0 or response_values.shape[1] == 0:
            raise MirtValidationError(
                "responses must contain at least one person and one item",
                parameter="responses",
                value=response_values.shape,
                expected="non-empty two-dimensional matrix",
            )

        if np.issubdtype(response_values.dtype, np.integer) or np.issubdtype(
            response_values.dtype, np.bool_
        ):
            observed = response_values >= 0
            invalid = observed & (response_values != 0) & (response_values != 1)
            response_numeric = response_values
        else:
            try:
                response_numeric = response_values.astype(np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    "responses must be numeric",
                    parameter="responses",
                    value=response_values,
                    expected="0, 1, or a negative/NaN missing value",
                ) from exc
            finite = np.isfinite(response_numeric)
            missing = np.isnan(response_numeric) | (finite & (response_numeric < 0.0))
            observed = finite & (response_numeric >= 0.0)
            invalid = (~missing & ~observed) | (
                observed & (response_numeric != 0.0) & (response_numeric != 1.0)
            )
        if np.any(invalid):
            raise MirtValidationError(
                "observed responses must be 0 or 1",
                parameter="responses",
                value=response_numeric[invalid],
                expected="0, 1, or a negative/NaN missing value",
            )
        response_integers = np.where(observed, response_numeric, -1).astype(np.int32)

        try:
            time_values = np.asarray(response_times, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "response_times must be numeric",
                parameter="response_times",
                value=response_times,
                expected="positive finite values or NaN",
            ) from exc
        if time_values.shape != response_values.shape:
            raise MirtValidationError(
                "response_times must have the same shape as responses",
                parameter="response_times",
                value=time_values.shape,
                expected=str(response_values.shape),
            )
        finite_times = np.isfinite(time_values)
        invalid_times = np.isinf(time_values) | (finite_times & (time_values <= 0.0))
        if np.any(invalid_times):
            raise MirtValidationError(
                "observed response_times must be finite and positive",
                parameter="response_times",
                value=time_values[invalid_times],
                expected="positive finite values or NaN",
            )
        if not np.any(observed) and not np.any(finite_times):
            raise MirtValidationError(
                "at least one accuracy or timing observation is required",
                parameter="responses",
                expected="at least one observed value",
            )

        log_rt = np.full(time_values.shape, np.nan, dtype=np.float64)
        np.log(time_values, out=log_rt, where=finite_times)
        return response_integers, log_rt

    def _chain_generators(self) -> list[np.random.Generator]:
        """Create reproducible independent random streams for every chain."""
        if self.n_chains == 1:
            return [np.random.default_rng(self.seed)]
        seed_sequence = np.random.SeedSequence(self.seed)
        return [
            np.random.default_rng(child) for child in seed_sequence.spawn(self.n_chains)
        ]

    def _run_chain(
        self,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        accuracy_model: Literal["2PL", "3PL"],
        rng: np.random.Generator,
        chain_idx: int,
    ) -> tuple[
        dict[str, NDArray[np.float64]],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Run one independent chain and return its retained draws."""
        n_persons, n_items = responses.shape
        n_samples = len(range(self.burnin, self.n_iter, self.thin))
        disc = np.ones(n_items)
        diff = np.zeros(n_items)
        guess = (
            np.full(n_items, min(0.2, 0.5 * self.priors.guess_upper))
            if accuracy_model == "3PL"
            else None
        )
        time_disc = np.ones(n_items)
        time_int = np.zeros(n_items)
        # The identification constraints fix mu = 0 and sigma[0, 0] = 1.
        mu = np.zeros(2)
        sigma = np.eye(2)
        theta = rng.standard_normal(n_persons)
        tau = rng.standard_normal(n_persons)

        chains = {
            "discrimination": np.zeros((n_samples, n_items)),
            "difficulty": np.zeros((n_samples, n_items)),
            "time_discrimination": np.zeros((n_samples, n_items)),
            "time_intensity": np.zeros((n_samples, n_items)),
            "mu_theta": np.zeros(n_samples),
            "mu_tau": np.zeros(n_samples),
            "sigma_11": np.zeros(n_samples),
            "sigma_22": np.zeros(n_samples),
            "sigma_12": np.zeros(n_samples),
        }
        if accuracy_model == "3PL":
            chains["guessing"] = np.zeros((n_samples, n_items))
        theta_samples = np.zeros((n_samples, n_persons))
        tau_samples = np.zeros((n_samples, n_persons))
        acceptance_counts = np.zeros(n_persons)
        current_proposal_sd = self.proposal_sd
        sample_idx = 0

        for iteration in range(self.n_iter):
            theta, tau, accepted = self._sample_person_params(
                responses,
                log_rt,
                theta,
                tau,
                disc,
                diff,
                guess,
                time_disc,
                time_int,
                mu,
                sigma,
                current_proposal_sd,
                rng,
            )
            acceptance_counts += accepted

            if (iteration + 1) % self.adapt_interval == 0 and iteration < self.burnin:
                accept_rate = np.mean(acceptance_counts) / self.adapt_interval
                if accept_rate < 0.2:
                    current_proposal_sd *= 0.8
                elif accept_rate > 0.5:
                    current_proposal_sd *= 1.2
                acceptance_counts[:] = 0

            disc, diff = self._sample_accuracy_params(
                responses, theta, disc, diff, rng, guess=guess
            )
            if guess is not None:
                guess = self._sample_guessing_params(
                    responses, theta, disc, diff, guess, rng
                )
            time_disc, time_int = self._sample_time_params(
                log_rt, tau, time_disc, time_int, rng
            )
            sigma = self._sample_population_params(theta, tau, rng)

            if iteration >= self.burnin and (iteration - self.burnin) % self.thin == 0:
                chains["discrimination"][sample_idx] = disc
                chains["difficulty"][sample_idx] = diff
                chains["time_discrimination"][sample_idx] = time_disc
                chains["time_intensity"][sample_idx] = time_int
                chains["mu_theta"][sample_idx] = mu[0]
                chains["mu_tau"][sample_idx] = mu[1]
                chains["sigma_11"][sample_idx] = sigma[0, 0]
                chains["sigma_22"][sample_idx] = sigma[1, 1]
                chains["sigma_12"][sample_idx] = sigma[0, 1]
                if accuracy_model == "3PL":
                    chains["guessing"][sample_idx] = guess
                theta_samples[sample_idx] = theta
                tau_samples[sample_idx] = tau
                sample_idx += 1

            if self.verbose and (iteration + 1) % 500 == 0:
                prefix = (
                    f"Chain {chain_idx + 1}/{self.n_chains}, "
                    if self.n_chains > 1
                    else ""
                )
                print(f"{prefix}iteration {iteration + 1}/{self.n_iter}")

        return chains, theta_samples, tau_samples

    def _sample_person_params(
        self,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        theta: NDArray[np.float64],
        tau: NDArray[np.float64],
        disc: NDArray[np.float64],
        diff: NDArray[np.float64],
        guess: NDArray[np.float64] | None,
        time_disc: NDArray[np.float64],
        time_int: NDArray[np.float64],
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        proposal_sd: float,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Sample (θ, τ) jointly via Metropolis-Hastings."""
        n_persons = len(theta)

        sigma_inv = np.linalg.inv(sigma)
        log_det_sigma = np.linalg.slogdet(sigma)[1]
        proposal_offsets = rng.normal(0.0, proposal_sd, size=(n_persons, 2))
        theta_proposed = theta + proposal_offsets[:, 0]
        tau_proposed = tau + proposal_offsets[:, 1]
        log_uniform = np.log(rng.random(n_persons))

        return rt_accept_person_proposals(
            responses,
            log_rt,
            theta,
            tau,
            theta_proposed,
            tau_proposed,
            log_uniform,
            disc,
            diff,
            time_disc,
            time_int,
            mu,
            sigma_inv,
            log_det_sigma,
            guess,
            use_rust=self.use_rust,
        )

    def _sample_accuracy_params(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        disc: NDArray[np.float64],
        diff: NDArray[np.float64],
        rng: np.random.Generator,
        guess: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update every item's ``(log a_j, b_j)`` by one random-walk MH step.

        Items are conditionally independent given ability, so all proposals
        are drawn and accepted together. The acceptance ratio uses the
        accuracy model's own likelihood: the 3PL with the current guessing
        parameters when ``guess`` is given, otherwise the 2PL.
        """
        n_items = len(disc)
        log_disc = np.log(disc)
        log_disc_prop = log_disc + rng.normal(0.0, _ACCURACY_PROPOSAL_SD, n_items)
        diff_prop = diff + rng.normal(0.0, _ACCURACY_PROPOSAL_SD, n_items)
        log_uniform = np.log(rng.random(n_items))
        disc_prop = np.exp(log_disc_prop)

        correct = responses == 1
        observed = responses >= 0
        log_like_curr = _accuracy_log_likelihoods(
            correct, observed, theta, disc, diff, (guess,)
        )[0]
        log_like_prop = _accuracy_log_likelihoods(
            correct, observed, theta, disc_prop, diff_prop, (guess,)
        )[0]

        priors = self.priors

        def log_prior(
            log_a: NDArray[np.float64], b: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            return (
                -0.5 * (log_a - priors.disc_mean) ** 2 / priors.disc_var
                - 0.5 * (b - priors.diff_mean) ** 2 / priors.diff_var
            )

        # The walk and the prior both live on log a, so no Jacobian term.
        log_accept = (log_like_prop + log_prior(log_disc_prop, diff_prop)) - (
            log_like_curr + log_prior(log_disc, diff)
        )
        accepted = log_uniform < log_accept
        return np.where(accepted, disc_prop, disc), np.where(accepted, diff_prop, diff)

    def _sample_guessing_params(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        disc: NDArray[np.float64],
        diff: NDArray[np.float64],
        guess: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Update every guessing parameter by one logit-scale MH step.

        With ``U = guess_upper`` the walk moves ``eta_j = logit(c_j / U)``,
        so proposals never leave ``(0, U)`` and the proposal is symmetric. The
        target on the ``eta`` scale is the 3PL likelihood times the
        ``Beta(guess_alpha, guess_beta)`` density of ``c_j`` times the
        Jacobian ``dc / deta = c (U - c) / U``.
        """
        n_items = len(guess)
        upper = self.priors.guess_upper
        alpha = self.priors.guess_alpha
        beta = self.priors.guess_beta
        log_upper = np.log(upper)

        def log_density(eta: NDArray[np.float64]) -> NDArray[np.float64]:
            log_guess = log_upper - np.logaddexp(0.0, -eta)
            log_gap = log_upper - np.logaddexp(0.0, eta)
            log_complement = np.log1p(-upper * expit(eta))
            # alpha * log c combines the prior's (alpha - 1) log c with the
            # Jacobian's log c; log_gap is the Jacobian's log(U - c).
            return alpha * log_guess + (beta - 1.0) * log_complement + log_gap

        eta = np.log(guess) - np.log(upper - guess)
        eta_prop = eta + rng.normal(0.0, _GUESSING_LOGIT_PROPOSAL_SD, n_items)
        guess_prop = upper * expit(eta_prop)
        log_uniform = np.log(rng.random(n_items))

        log_like_curr, log_like_prop = _accuracy_log_likelihoods(
            responses == 1,
            responses >= 0,
            theta,
            disc,
            diff,
            (guess, guess_prop),
        )
        log_accept = (log_like_prop + log_density(eta_prop)) - (
            log_like_curr + log_density(eta)
        )
        # Rejecting |eta| >= 30 truncates the support to guessing values that
        # stay distinguishable from 0 and U in floating point.
        accepted = (log_uniform < log_accept) & (
            np.abs(eta_prop) < _GUESSING_LOGIT_LIMIT
        )
        return np.where(accepted, guess_prop, guess)

    def _sample_time_params(
        self,
        log_rt: NDArray[np.float64],
        tau: NDArray[np.float64],
        time_disc: NDArray[np.float64],
        time_int: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sample the lognormal time item parameters for every item at once.

        Given speed ``tau`` the shifted log times ``log t_ij + tau_i`` are
        ``N(beta_j, 1 / alpha_j**2)``. Time intensity ``beta_j`` gets an exact
        conjugate normal draw. Log time discrimination ``eta_j = log alpha_j``
        has the non-conjugate ``N(time_disc_mean, time_disc_var)`` prior and is
        updated by a few random-walk Metropolis steps whose scale follows the
        conditional posterior precision ``2 n_j + 1 / time_disc_var``. The
        conditional depends on the data only through ``n_j`` and the residual
        sum of squares, so the extra steps cost ``O(n_items)`` each.
        """
        valid = ~np.isnan(log_rt)
        n_valid = np.count_nonzero(valid, axis=0)
        shifted = np.where(valid, log_rt + tau[:, None], 0.0)

        intensity_var = self.priors.time_int_var
        precision = time_disc**2
        post_var = 1.0 / (1.0 / intensity_var + n_valid * precision)
        post_mean = post_var * (
            self.priors.time_int_mean / intensity_var
            + precision * np.sum(shifted, axis=0)
        )
        new_time_int = post_mean + np.sqrt(post_var) * rng.standard_normal(
            time_int.shape
        )

        residuals = np.where(valid, shifted - new_time_int, 0.0)
        half_ss = 0.5 * np.sum(residuals**2, axis=0)
        prior_mean = self.priors.time_disc_mean
        prior_var = self.priors.time_disc_var

        def log_conditional(eta: NDArray[np.float64]) -> NDArray[np.float64]:
            return (
                n_valid * eta
                - half_ss * np.exp(2.0 * eta)
                - 0.5 * (eta - prior_mean) ** 2 / prior_var
            )

        draw_shape = (_TIME_DISCRIMINATION_MH_STEPS,) + time_disc.shape
        step = 2.4 / np.sqrt(2.0 * n_valid + 1.0 / prior_var)
        offsets = step * rng.standard_normal(draw_shape)
        log_uniform = np.log(rng.random(draw_shape))
        eta_initial = np.log(time_disc)
        eta = eta_initial
        log_target = log_conditional(eta)
        for offset, threshold in zip(offsets, log_uniform, strict=True):
            proposal = eta + offset
            proposal_target = log_conditional(proposal)
            accepted = threshold < proposal_target - log_target
            eta = np.where(accepted, proposal, eta)
            log_target = np.where(accepted, proposal_target, log_target)
        # Items that rejected every proposal keep their exact previous value.
        new_time_disc = np.where(eta == eta_initial, time_disc, np.exp(eta))
        return new_time_disc, new_time_int

    def _sample_population_params(
        self,
        theta: NDArray[np.float64],
        tau: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw the ability-speed covariance under the identification.

        The mean is fixed at zero and ``Sigma[0, 0] = Var(theta)`` at one, so
        only ``Cov(theta, tau)`` and ``Var(tau)`` are random. They are drawn
        exactly from the ``IW(sigma_df, sigma_scale)`` prior conditioned on
        the unit ability variance: the conjugate regression of speed on
        ability.
        """
        return sample_unit_leading_covariance(
            np.column_stack([theta, tau]),
            self.priors.sigma_df,
            self.priors.sigma_scale,
            rng,
        )

    def _compute_dic(
        self,
        model: ResponseTimeModel,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        theta_samples: NDArray[np.float64],
        tau_samples: NDArray[np.float64],
        log_likes: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute Deviance Information Criterion."""
        if log_likes is None:
            log_likes = self._posterior_log_likelihoods(
                model, responses, log_rt, theta_samples, tau_samples
            )
        deviances = -2.0 * np.sum(log_likes, axis=1)
        d_bar = np.mean(deviances)

        theta_mean = np.mean(theta_samples, axis=0)
        tau_mean = np.mean(tau_samples, axis=0)
        ll_at_mean = np.sum(
            model.joint_log_likelihood(responses, log_rt, theta_mean, tau_mean)
        )
        d_theta_bar = -2 * ll_at_mean

        p_d = d_bar - d_theta_bar
        dic = d_bar + p_d

        return float(dic)

    def _compute_waic(
        self,
        model: ResponseTimeModel,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        theta_samples: NDArray[np.float64],
        tau_samples: NDArray[np.float64],
        log_likes: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute Watanabe-Akaike Information Criterion."""
        if log_likes is None:
            log_likes = self._posterior_log_likelihoods(
                model, responses, log_rt, theta_samples, tau_samples
            )
        n_samples = log_likes.shape[0]

        lppd = np.sum(logsumexp(log_likes, axis=0) - np.log(n_samples))
        p_waic = np.sum(np.var(log_likes, axis=0, ddof=1)) if n_samples > 1 else 0.0

        waic = -2 * (lppd - p_waic)

        return float(waic)

    def _draw_log_likelihoods(
        self,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        theta_samples: NDArray[np.float64],
        tau_samples: NDArray[np.float64],
        chains: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Evaluate every retained draw with its own iteration's parameters.

        DIC and WAIC average the likelihood over the joint posterior, so each
        ability-speed draw is paired with the item parameters stored at the
        same iteration rather than with their posterior means.
        """
        guessing = chains.get("guessing")
        log_likes = np.empty(theta_samples.shape, dtype=np.float64)
        for sample_idx in range(theta_samples.shape[0]):
            log_likes[sample_idx] = rt_joint_log_likelihood(
                responses,
                log_rt,
                theta_samples[sample_idx],
                tau_samples[sample_idx],
                chains["discrimination"][sample_idx],
                chains["difficulty"][sample_idx],
                chains["time_discrimination"][sample_idx],
                chains["time_intensity"][sample_idx],
                None if guessing is None else guessing[sample_idx],
                use_rust=self.use_rust,
            )
        return log_likes

    @staticmethod
    def _posterior_log_likelihoods(
        model: ResponseTimeModel,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        theta_samples: NDArray[np.float64],
        tau_samples: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate every retained person-level log likelihood once."""
        n_samples = theta_samples.shape[0]
        batched_likelihood = getattr(model, "joint_log_likelihood_samples", None)
        if callable(batched_likelihood):
            log_likes = np.asarray(
                batched_likelihood(
                    responses,
                    log_rt,
                    theta_samples,
                    tau_samples,
                ),
                dtype=np.float64,
            )
            expected_shape = (n_samples, responses.shape[0])
            if log_likes.shape != expected_shape:
                raise ValueError(
                    "joint_log_likelihood_samples must return shape "
                    f"{expected_shape}, got {log_likes.shape}"
                )
            return log_likes

        log_likes = np.empty((n_samples, responses.shape[0]), dtype=np.float64)
        for sample_idx in range(n_samples):
            log_likes[sample_idx] = model.joint_log_likelihood(
                responses,
                log_rt,
                theta_samples[sample_idx],
                tau_samples[sample_idx],
            )
        return log_likes

    def _compute_rhat(self, chains: dict[str, NDArray[np.float64]]) -> dict[str, float]:
        """Compute split-chain Gelman-Rubin convergence diagnostics."""
        return {
            name: self._rhat_for_array(np.asarray(samples, dtype=np.float64))
            for name, samples in chains.items()
        }

    @staticmethod
    def _rhat_for_array(samples: NDArray[np.float64]) -> float:
        """Compute split R-hat averaged across trailing parameter dimensions."""
        if samples.ndim < 2:
            raise ValueError("diagnostic samples must have chain and draw dimensions")
        n_chains, n_draws = samples.shape[:2]
        if n_draws < 4:
            return float("nan")

        half = n_draws // 2
        flat = samples.reshape(n_chains, n_draws, -1)
        split = np.concatenate((flat[:, :half], flat[:, -half:]), axis=0)
        chain_means = np.mean(split, axis=1)
        within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
        between = half * np.var(chain_means, axis=0, ddof=1)
        variance = ((half - 1) / half) * within + between / half

        values = np.ones_like(within)
        varying = within > 0.0
        values[varying] = np.sqrt(variance[varying] / within[varying])
        values[(~varying) & (between > 0.0)] = np.inf
        return float(np.mean(values))

    def _compute_ess(self, chains: dict[str, NDArray[np.float64]]) -> dict[str, float]:
        """Compute effective sample sizes using FFT autocorrelations."""
        return {
            name: self._ess_for_array(np.asarray(samples, dtype=np.float64))
            for name, samples in chains.items()
        }

    @staticmethod
    def _ess_for_array(samples: NDArray[np.float64]) -> float:
        """Compute ESS averaged across trailing parameter dimensions."""
        if samples.ndim < 2:
            raise ValueError("diagnostic samples must have chain and draw dimensions")
        n_chains, n_draws = samples.shape[:2]
        total_draws = n_chains * n_draws
        if n_draws < 4:
            return float(total_draws)

        flat = samples.reshape(n_chains, n_draws, -1)
        centered = flat - np.mean(flat, axis=1, keepdims=True)
        fft_size = 1 << (2 * n_draws - 1).bit_length()
        transformed = np.fft.rfft(centered, n=fft_size, axis=1)
        autocovariance = np.fft.irfft(
            transformed * np.conjugate(transformed), n=fft_size, axis=1
        )[:, :n_draws]
        autocovariance /= np.arange(n_draws, 0, -1)[None, :, None]

        variances = autocovariance[:, 0]
        valid = variances > 0.0
        correlations = np.divide(
            autocovariance,
            variances[:, None, :],
            out=np.zeros_like(autocovariance),
            where=valid[:, None, :],
        )
        valid_counts = np.sum(valid, axis=0)
        mean_correlations = np.divide(
            np.sum(correlations, axis=0),
            valid_counts[None, :],
            out=np.zeros_like(correlations[0]),
            where=valid_counts[None, :] > 0,
        )

        integrated_time = np.ones(flat.shape[2])
        active = valid_counts > 0
        for lag in range(1, n_draws - 1, 2):
            pair = mean_correlations[lag] + mean_correlations[lag + 1]
            active &= pair > 0.0
            integrated_time[active] += 2.0 * pair[active]

        effective = total_draws / integrated_time
        effective[valid_counts == 0] = total_draws
        return float(np.mean(np.minimum(effective, total_draws)))
