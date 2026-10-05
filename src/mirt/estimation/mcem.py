"""Monte Carlo EM (MCEM) and Quasi-Monte Carlo EM (QMCEM) estimators.

These estimation methods are useful for high-dimensional IRT models where
Gauss-Hermite quadrature becomes computationally infeasible. They use
Monte Carlo integration in the E-step instead of numerical quadrature.

MCEM uses standard pseudo-random sampling, while QMCEM uses low-discrepancy
sequences (Quasi-Monte Carlo) for more uniform coverage of the integration
space and faster convergence.

References:
    Wei, G. C., & Tanner, M. A. (1990). A Monte Carlo implementation of the
        EM algorithm and the poor man's data augmentation algorithms.
        Journal of the American Statistical Association, 85(411), 699-704.

    Cagnone, S., & Monari, P. (2013). Latent variable models for ordinal
        data by using the adaptive quadrature approximation.
        Computational Statistics, 28(2), 597-619.

    Caffo, B. S., Jank, W., & Jones, G. L. (2005). Ascent-based Monte Carlo
        expectation-maximization. Journal of the Royal Statistical Society:
        Series B, 67(2), 235-251.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import xlog1py, xlogy
from scipy.stats import qmc

from mirt._model_defaults import (
    original_model_hook,
    uses_builtin_model_hooks,
    uses_original_model_hook,
)
from mirt.constants import PROB_EPSILON
from mirt.estimation._acceleration import FreeItemParameters
from mirt.estimation._gaussian_kernel import gaussian_log_kernel
from mirt.estimation._mc_likelihood import (
    sampled_log_likelihoods,
    uses_default_sample_likelihood,
)
from mirt.estimation._posterior import normalize_log_posterior
from mirt.estimation.base import (
    BaseEstimator,
    StartValues,
    _apply_starting_values,
    _validate_start,
)
from mirt.models.base import DichotomousItemModel, PolytomousItemModel
from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


_MAX_LIKELIHOOD_ELEMENTS = 2_000_000
_MAX_QMC_COUNT_ELEMENTS = 1_000_000
# Ascent-based MCEM (Caffo, Jank and Jones, 2005): one-sided 75% normal
# quantile for the confidence bounds on the log-likelihood change, the factor
# by which the Monte Carlo sample grows when a step is not a confirmed ascent,
# and the number of unconfirmed steps at ``max_samples`` before stopping.
_ASCENT_Z = 0.6744897501960817
_SAMPLE_GROWTH = 1.5
_STALLED_ITERATIONS = 3


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    """Return a validated integer count."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) < minimum
    ):
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return int(value)


def _boolean(value: bool, name: str) -> bool:
    """Return a validated Boolean option."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _seed_value(seed: int | None) -> int | None:
    """Return a backend-safe non-negative random seed."""
    if seed is None:
        return None
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, Integral)
        or int(seed) < 0
    ):
        raise ValueError("seed must be a non-negative integer or None")
    return int(seed)


def _validated_prior(
    prior_mean: NDArray[np.float64] | None,
    prior_cov: NDArray[np.float64] | None,
    n_factors: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate a finite Gaussian prior and return its Cholesky factor."""
    mean = (
        np.zeros(n_factors, dtype=np.float64)
        if prior_mean is None
        else np.asarray(prior_mean, dtype=np.float64)
    )
    if mean.shape != (n_factors,):
        raise ValueError(f"prior_mean must have shape ({n_factors},)")
    if not np.all(np.isfinite(mean)):
        raise ValueError("prior_mean must contain only finite values")

    covariance = (
        np.eye(n_factors, dtype=np.float64)
        if prior_cov is None
        else np.asarray(prior_cov, dtype=np.float64)
    )
    expected_shape = (n_factors, n_factors)
    if covariance.shape != expected_shape:
        raise ValueError(f"prior_cov must have shape {expected_shape}")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("prior_cov must contain only finite values")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("prior_cov must be symmetric")
    with np.errstate(over="ignore"):
        symmetric = (covariance + covariance.T) * 0.5
    overflow = ~np.isfinite(symmetric)
    if np.any(overflow):
        symmetric[overflow] = 0.5 * covariance[overflow] + 0.5 * covariance.T[overflow]
    covariance = symmetric
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("prior_cov must be positive definite") from exc
    return mean.copy(), cholesky


def _log_likelihood_change(
    weights: NDArray[np.float64],
    base: NDArray[np.float64],
    other: NDArray[np.float64],
) -> tuple[float, float]:
    """Estimate ``log p(y; other) - log p(y; base)`` from common draws.

    ``weights`` are normalized posterior weights of each person's draws under
    the base parameters, and ``base`` and ``other`` hold the draws'
    log-likelihoods under the two parameter sets. Each person contributes the
    log of the weighted mean likelihood ratio. Returns the estimate and its
    delta-method Monte Carlo standard error.
    """
    log_ratio = other - base
    shift = np.max(log_ratio, axis=1, keepdims=True)
    ratio = np.exp(log_ratio - shift)
    mean_ratio = np.sum(weights * ratio, axis=1)
    change = float(np.sum(np.log(mean_ratio) + shift[:, 0]))
    ratio -= mean_ratio[:, None]
    ratio *= weights
    variance = np.sum(ratio * ratio, axis=1) / (mean_ratio * mean_ratio)
    return change, float(np.sqrt(np.sum(variance)))


class MCEMEstimator(BaseEstimator):
    """Monte Carlo EM estimator for IRT models.

    Uses Monte Carlo integration in the E-step, making it suitable for
    models with many latent dimensions where quadrature is infeasible.

    Parameters
    ----------
    n_samples : int
        Initial number of Monte Carlo samples per person per iteration.
        More samples give more accurate E-step but slower computation.
    max_iter : int
        Maximum number of EM iterations.
    tol : float, default=1e-3
        Convergence tolerance for the change in marginal log-likelihood
        between iterates. MCEM converges when a 50% confidence interval for
        the change lies within ``(-tol, tol)``; see Notes.
    verbose : bool
        Whether to print progress.
    seed : int or None
        Random seed for reproducibility.
    importance_sampling : bool
        Whether to use importance sampling from the prior.
        Improves efficiency when posterior differs from prior.
    max_samples : int, optional
        Largest number of samples per person that sample-size growth may
        reach. Defaults to ten times ``n_samples``. Draws and their
        likelihoods take memory proportional to persons times samples.

    Notes
    -----
    MCEM is particularly useful when:
    - The model has more than 3-4 latent dimensions
    - Quadrature-based EM is too slow
    - Exact integration is not required

    Convergence adapts ascent-based MCEM (Caffo, Jank and Jones, 2005).
    From the second iteration on, the change in marginal log-likelihood made
    by the previous M-step is estimated on the fresh draws, which are shared
    by both parameter sets, together with its delta-method Monte Carlo
    standard error. A change whose lower 75% confidence bound is not positive
    is within Monte Carlo error. Unlike the original algorithm, which repeats
    such a step with more draws, the step is kept and the next iteration's
    sample grows by half, up to ``max_samples``. The fit converges once the
    75% bounds lie within ``(-tol, tol)``. Once ``max_samples`` is reached, three consecutive changes
    within Monte Carlo error stop the fit early with ``converged=False`` and a
    warning, because more iterations cannot improve the precision.
    When ``max_iter`` is reached, the last M-step is judged by the change in
    the log-likelihood estimate on its own draws. ``sample_size_history``
    records the sample size of every iteration.

    Built-in logistic, affine, and polytomous items use shared analytic
    gradients. Small item sample blocks are prepared once; larger draws
    stream bounded blocks without copying every observed person's samples.
    Custom models and item objectives retain numerical optimization.
    Ordinary sampled likelihoods reduce bounded public probability blocks
    against unexpanded responses. Custom likelihood and validation overrides
    retain their model-based evaluation path.

    Set ``compute_standard_errors=True`` to estimate approximate diagonal
    complete-data standard errors with the final draws and weights held fixed.
    This excludes missing information, parameter covariances, and sampling
    uncertainty. Built-in polytomous items use exact diagonal curvature;
    logistic and affine items differentiate prepared gradients.
    ``se_step_size`` controls finite differences (default 1e-5) and does not
    affect exact polytomous curvature.
    Standard errors remain disabled by default to preserve sampling costs.
    """

    _minimum_samples = 50
    # QMCEM's shared grid makes its iterations deterministic, and stochastic EM
    # keeps a fixed number of chains, so both use the plain change rule.
    _adaptive_sample_size = True

    def __init__(
        self,
        n_samples: int = 500,
        max_iter: int = 500,
        tol: float = 1e-3,
        verbose: bool = False,
        seed: int | None = None,
        importance_sampling: bool = True,
        compute_standard_errors: bool = False,
        se_step_size: float = 1e-5,
        max_samples: int | None = None,
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        self.n_samples = _positive_integer(
            n_samples, "n_samples", minimum=self._minimum_samples
        )
        self.max_samples = (
            10 * self.n_samples
            if max_samples is None
            else _positive_integer(max_samples, "max_samples", minimum=self.n_samples)
        )
        self._sample_size_history: list[int] = []
        self.seed = _seed_value(seed)
        self.importance_sampling = _boolean(importance_sampling, "importance_sampling")
        self.compute_standard_errors = _boolean(
            compute_standard_errors, "compute_standard_errors"
        )
        if (
            isinstance(se_step_size, (bool, np.bool_))
            or not isinstance(se_step_size, Real)
            or not np.isfinite(se_step_size)
            or se_step_size <= 0.0
        ):
            raise ValueError("se_step_size must be a finite positive number")
        self.se_step_size: float = float(se_step_size)
        self._rng: np.random.Generator | None = None

    @property
    def sample_size_history(self) -> list[int]:
        """Monte Carlo samples per person used by each iteration of the last fit."""
        return self._sample_size_history.copy()

    def _random_generator(self) -> np.random.Generator:
        """Return the initialized fit-local random generator."""
        if self._rng is None:
            raise RuntimeError("the estimator random generator is not initialized")
        return self._rng

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
        *,
        start: StartValues = "default",
    ) -> FitResult:
        """Fit model using Monte Carlo EM algorithm.

        Parameters
        ----------
        model : BaseItemModel
            IRT model to fit. Coordinates fixed with
            ``set_free_parameter_masks`` keep their values.
        responses : ndarray of shape (n_persons, n_items)
            Response matrix
        prior_mean : ndarray of shape (n_factors,), optional
            Prior mean for latent abilities
        prior_cov : ndarray of shape (n_factors, n_factors), optional
            Prior covariance for latent abilities
        start : {"default", "model"} or mapping, default="default"
            Starting values, as for :meth:`EMEstimator.fit`.

        Returns
        -------
        FitResult
            Fitted model with estimates and diagnostics

        Raises
        ------
        MirtModelError
            If the model has free parameters shared by all items, such as
            rating-scale thresholds, which the itemwise M-step cannot update.
        """
        from mirt.results.fit_result import FitResult

        start = _validate_start(start)
        responses = self._validate_responses(responses, model.n_items)
        self._check_shared_parameters(model)
        n_persons = responses.shape[0]
        n_factors = model.n_factors

        self._rng = np.random.default_rng(self.seed)

        prior_mean, cholesky = _validated_prior(prior_mean, prior_cov, n_factors)

        _apply_starting_values(model, start)

        adaptive = self._adaptive_sample_size
        free_parameters = FreeItemParameters(model) if adaptive else None
        configured_samples = next_samples = self.n_samples
        self._convergence_history = []
        self._sample_size_history = []
        prev_ll = -np.inf
        previous: NDArray[np.float64] | None = None
        converged = False
        stalled = 0

        try:
            for iteration in range(self.max_iter):
                self.n_samples = next_samples
                theta_samples, weights, current_ll = self._e_step_and_marginal_ll(
                    model, responses, prior_mean, cholesky, n_factors
                )
                self._convergence_history.append(current_ll)
                self._sample_size_history.append(self.n_samples)

                self._log_iteration(iteration, current_ll)

                if free_parameters is None:
                    converged = self._check_convergence(prev_ll, current_ll)
                elif previous is not None:
                    change, error = self._iterate_change(
                        model,
                        responses,
                        theta_samples,
                        weights,
                        free_parameters,
                        previous,
                    )
                    converged = self._monte_carlo_converged(change, error)
                    if change - _ASCENT_Z * error > 0.0:
                        stalled = 0
                    elif self.n_samples < self.max_samples:
                        next_samples = min(
                            math.ceil(_SAMPLE_GROWTH * self.n_samples),
                            self.max_samples,
                        )
                    else:
                        stalled += 1

                if converged:
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break
                if stalled >= _STALLED_ITERATIONS:
                    warnings.warn(
                        "MCEM stopped before convergence: log-likelihood changes "
                        f"stayed within Monte Carlo error at max_samples="
                        f"{self.max_samples}. Increase max_samples or tol.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break

                prev_ll = current_ll
                if free_parameters is not None:
                    previous = free_parameters.get(model)

                self._m_step_mc(model, responses, theta_samples, weights)
                if iteration + 1 < self.max_iter:
                    # Release the previous draw before allocating its replacement.
                    del theta_samples, weights
            else:
                current_ll, weights = self._refresh_mc_state(
                    model, responses, theta_samples, weights
                )
                self._convergence_history.append(current_ll)
                # Both estimates use the last draw, so they share its noise.
                converged = self._check_convergence(prev_ll, current_ll)

            model._is_fitted = True

            standard_errors = self._compute_standard_errors_mc(
                model, responses, theta_samples, weights
            )
        finally:
            self.n_samples = configured_samples

        n_params = model.n_parameters
        aic = self._compute_aic(current_ll, n_params)
        bic = self._compute_bic(current_ll, n_params, n_persons)

        return FitResult(
            model=model,
            log_likelihood=current_ll,
            n_iterations=iteration + 1,
            converged=converged,
            standard_errors=standard_errors,
            aic=aic,
            bic=bic,
            n_observations=n_persons,
            n_parameters=n_params,
        )

    def _iterate_change(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
        free_parameters: FreeItemParameters,
        previous: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Estimate the log-likelihood gain over ``previous`` on the current draws.

        Returns the estimated change from the previous iterate to the current
        parameters and its Monte Carlo standard error.
        """
        current_values = self._sample_log_likelihoods(model, responses, theta_samples)
        current = free_parameters.get(model)
        if not free_parameters.set(model, previous):
            raise ValueError("the previous MCEM iterate is not a valid parameter set")
        try:
            previous_values = self._sample_log_likelihoods(
                model, responses, theta_samples
            )
        finally:
            free_parameters.set(model, current)
        loss, error = _log_likelihood_change(weights, current_values, previous_values)
        return -loss, error

    def _monte_carlo_converged(self, change: float, error: float) -> bool:
        """Return whether a confidence interval for the change is within ``tol``."""
        return abs(change) + _ASCENT_Z * error < self.tol

    @staticmethod
    def _validated_log_likelihoods(
        values: NDArray[np.float64],
        expected_shape: tuple[int, int],
    ) -> NDArray[np.float64]:
        """Validate a person-by-sample log-likelihood matrix."""
        log_likelihoods = np.asarray(values, dtype=np.float64)
        if log_likelihoods.shape != expected_shape:
            raise ValueError(
                "model log-likelihood output has shape "
                f"{log_likelihoods.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(log_likelihoods)):
            raise ValueError("model log-likelihood output must be finite")
        return log_likelihoods

    def _sample_log_likelihoods(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate person-specific samples in memory-bounded batches."""
        n_persons = responses.shape[0]
        expected_shape = (n_persons, self.n_samples, model.n_factors)
        samples = np.asarray(theta_samples)
        if samples.shape != expected_shape:
            raise ValueError(f"theta_samples must have shape {expected_shape}")
        prepared = sampled_log_likelihoods(model, responses, samples)
        if prepared is not None:
            return prepared
        samples = np.asarray(samples, dtype=np.float64)
        if not np.all(np.isfinite(samples)):
            raise ValueError("theta_samples must contain only finite values")

        log_likelihoods = np.empty((n_persons, self.n_samples), dtype=np.float64)
        elements_per_sample = max(1, n_persons * model.n_items)
        chunk_size = max(
            1,
            min(self.n_samples, _MAX_LIKELIHOOD_ELEMENTS // elements_per_sample),
        )
        for start in range(0, self.n_samples, chunk_size):
            stop = min(start + chunk_size, self.n_samples)
            width = stop - start
            theta_chunk = samples[:, start:stop, :].reshape(
                n_persons * width, model.n_factors
            )
            response_chunk = np.repeat(responses, width, axis=0)
            values = np.asarray(
                model.log_likelihood(response_chunk, theta_chunk), dtype=np.float64
            )
            if values.shape != (n_persons * width,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    "model.log_likelihood() returned invalid sampled values"
                )
            log_likelihoods[:, start:stop] = values.reshape(n_persons, width)
        return log_likelihoods

    @staticmethod
    def _normalized_importance_weights(
        log_likelihoods: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalize prior-sample likelihood ratios by person."""
        # Custom callers may supply views or cache a read-only likelihood array.
        owned = np.array(log_likelihoods, dtype=np.float64, copy=True)
        weights, normalizer = normalize_log_posterior(owned)
        return weights, normalizer[:, None]

    @staticmethod
    def _gaussian_log_kernel(
        theta_samples: NDArray[np.float64],
        prior_mean: NDArray[np.float64],
        cholesky: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate a Gaussian prior up to its shared normalizing constant."""
        return gaussian_log_kernel(theta_samples, prior_mean, cholesky)

    def _draw_posterior_samples(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        cholesky: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Draw parallel Metropolis samples from every person posterior."""
        samples, weights, _ = self._draw_posterior_state(
            model, responses, prior_mean, cholesky, n_factors
        )
        return samples, weights

    def _draw_posterior_state(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        cholesky: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return posterior draws, weights, and their final likelihood values."""
        n_persons = responses.shape[0]
        rng = self._random_generator()
        standard_normal = rng.standard_normal((n_persons, self.n_samples, n_factors))
        current = np.einsum("ij,...j->...i", cholesky, standard_normal)
        current += prior_mean
        del standard_normal
        expected_shape = (n_persons, self.n_samples)
        # Custom callbacks may return cached views or read-only buffers, or
        # share scratch between likelihood and prior evaluation.
        current_ll = np.array(
            self._validated_log_likelihoods(
                self._sample_log_likelihoods(model, responses, current), expected_shape
            ),
            copy=True,
        )
        current_lp = np.array(
            self._gaussian_log_kernel(current, prior_mean, cholesky),
            dtype=np.float64,
            copy=True,
        )
        if current_lp.shape != expected_shape:
            raise ValueError(f"Gaussian log kernel must have shape {expected_shape}")

        proposal_scale = 0.5
        proposal_ll = np.empty(expected_shape)
        for _ in range(20):
            standard_proposal = rng.standard_normal(current.shape)
            proposal = np.einsum("ij,...j->...i", cholesky, standard_proposal)
            del standard_proposal
            proposal *= proposal_scale
            proposal += current
            np.copyto(
                proposal_ll,
                self._validated_log_likelihoods(
                    self._sample_log_likelihoods(model, responses, proposal),
                    expected_shape,
                ),
            )
            proposal_lp = np.asarray(
                self._gaussian_log_kernel(proposal, prior_mean, cholesky),
                dtype=np.float64,
            )
            if proposal_lp.shape != expected_shape:
                raise ValueError(
                    f"Gaussian log kernel must have shape {expected_shape}"
                )
            log_acceptance = proposal_ll - current_ll
            log_acceptance += proposal_lp - current_lp
            uniforms = np.maximum(
                rng.random((n_persons, self.n_samples)),
                np.nextafter(0.0, 1.0),
            )
            accepted = np.log(uniforms) < log_acceptance
            np.copyto(current, proposal, where=accepted[:, :, None])
            np.copyto(current_ll, proposal_ll, where=accepted)
            np.copyto(current_lp, proposal_lp, where=accepted)
            del proposal, proposal_lp, log_acceptance, uniforms, accepted

        weights = np.full(
            (n_persons, self.n_samples),
            1.0 / self.n_samples,
            dtype=np.float64,
        )
        return current, weights, current_ll

    def _e_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        L: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """E-step using Monte Carlo sampling.

        Returns theta samples and their importance weights for each person.
        """
        if not self.importance_sampling:
            return self._draw_posterior_samples(
                model, responses, prior_mean, L, n_factors
            )
        samples, weights, _ = self._e_step_mc_state(
            model, responses, prior_mean, L, n_factors
        )
        return samples, weights

    def _e_step_mc_state(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        L: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return draws, weights, and importance normalizers or posterior likelihoods."""
        if not self.importance_sampling:
            return self._draw_posterior_state(
                model, responses, prior_mean, L, n_factors
            )
        n_persons = responses.shape[0]
        rng = self._random_generator()
        z = rng.standard_normal((n_persons, self.n_samples, n_factors))
        theta_samples = np.einsum("ij,...j->...i", L, z)
        theta_samples += prior_mean
        del z
        log_likes = self._sample_log_likelihoods(model, responses, theta_samples)
        weights, log_normalizer = self._normalized_importance_weights(log_likes)
        return theta_samples, weights, log_normalizer

    def _uses_default_sampling_methods(self, model: BaseItemModel) -> bool:
        """Reuse E-step evidence only when existing sampling hooks are unchanged."""
        methods = _DEFAULT_MC_STATE_METHODS.get(type(self))
        return (
            methods is not None
            and uses_default_sample_likelihood(model)
            and (
                type(self) is not QMCEMEstimator
                or (
                    uses_original_model_hook(model, "log_likelihood_batch")
                    and type(model).log_likelihood_batch
                    in _DEFAULT_QMC_BATCH_LIKELIHOODS
                )
            )
            and MCEMEstimator._sample_log_likelihoods is _DEFAULT_MC_SAMPLE_LIKELIHOODS
            and all(
                name not in vars(self) and getattr(type(self), name) is method
                for name, method in methods.items()
            )
        )

    def _e_step_and_marginal_ll(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        cholesky: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Evaluate one E-step, reusing fresh evidence for ordinary fit reporting."""
        if not self._uses_default_sampling_methods(model):
            samples, weights = self._e_step_mc(
                model, responses, prior_mean, cholesky, n_factors
            )
            marginal_ll = self._estimate_marginal_ll(model, responses, samples, weights)
            return samples, weights, marginal_ll
        samples, weights, evidence = self._e_step_mc_state(
            model, responses, prior_mean, cholesky, n_factors
        )
        if self.importance_sampling:
            log_marginal = evidence.ravel() - np.log(self.n_samples)
        else:
            log_marginal = -(logsumexp(-evidence, axis=1) - np.log(self.n_samples))
        return samples, weights, float(np.sum(log_marginal))

    def _estimate_marginal_ll(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Estimate the marginal log-likelihood on the current samples."""
        marginal_ll, _ = self._refresh_mc_state(
            model, responses, theta_samples, weights
        )
        return marginal_ll

    def _refresh_mc_state(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        """Evaluate current parameters on an existing Monte Carlo draw."""
        log_likes = self._sample_log_likelihoods(model, responses, theta_samples)
        if self.importance_sampling:
            weights, log_normalizer = self._normalized_importance_weights(log_likes)
            log_marginal = log_normalizer.ravel() - np.log(self.n_samples)
        else:
            # Posterior draws satisfy E[1 / p(y | theta)] = 1 / p(y).
            inverse_normalizer = logsumexp(-log_likes, axis=1, keepdims=False) - np.log(
                self.n_samples
            )
            log_marginal = -inverse_normalizer

        return float(np.sum(log_marginal)), weights

    def _m_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """M-step: optimize item parameters using weighted samples."""
        n_items = model.n_items

        for item_idx in range(n_items):
            self._optimize_item_mc(model, item_idx, responses, theta_samples, weights)

    def _item_expected_log_likelihood(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Return a chunked weighted log-likelihood for one observed item."""
        n_persons = len(responses)
        if theta_samples.shape != (
            n_persons,
            self.n_samples,
            model.n_factors,
        ):
            raise ValueError("theta_samples has an incompatible shape")
        if weights.shape != (n_persons, self.n_samples):
            raise ValueError("weights has an incompatible shape")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and non-negative")

        elements_per_sample = max(1, n_persons * model.n_items)
        chunk_size = max(
            1,
            min(self.n_samples, _MAX_LIKELIHOOD_ELEMENTS // elements_per_sample),
        )
        total = 0.0
        for start in range(0, self.n_samples, chunk_size):
            stop = min(start + chunk_size, self.n_samples)
            width = stop - start
            theta_chunk = theta_samples[:, start:stop, :].reshape(
                n_persons * width, model.n_factors
            )
            response_chunk = np.repeat(responses, width).astype(np.intp, copy=False)
            weight_chunk = weights[:, start:stop].reshape(-1)
            probabilities = np.asarray(
                model.probability(theta_chunk, item_idx), dtype=np.float64
            )

            if model.is_polytomous:
                if (
                    probabilities.ndim != 2
                    or probabilities.shape[0] != n_persons * width
                    or np.any(response_chunk >= probabilities.shape[1])
                ):
                    raise ValueError(
                        "model returned invalid item category probabilities"
                    )
                selected = probabilities[np.arange(n_persons * width), response_chunk]
                if not np.all(np.isfinite(selected)):
                    raise ValueError("model returned non-finite item probabilities")
                log_probability = np.log(np.clip(selected, PROB_EPSILON, 1.0))
            else:
                probabilities = probabilities.reshape(-1)
                if probabilities.shape != (n_persons * width,) or not np.all(
                    np.isfinite(probabilities)
                ):
                    raise ValueError("model returned invalid item probabilities")
                probabilities = np.clip(probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON)
                log_probability = response_chunk * np.log(probabilities) + (
                    1 - response_chunk
                ) * np.log1p(-probabilities)
            total += float(weight_chunk @ log_probability)
        return total

    def _optimize_item_mc(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Optimize parameters for a single item using MC samples."""
        item_responses = responses[:, item_idx]
        valid_mask = item_responses >= 0

        if not valid_mask.any():
            return

        current_params, bounds = self._get_item_params_and_bounds(model, item_idx)
        if current_params.size == 0:
            return

        if self._uses_default_item_methods():
            from mirt.estimation._mc_objective import prepare_mc_objective

            prepared = prepare_mc_objective(
                model,
                item_idx,
                item_responses,
                theta_samples,
                weights,
                self.n_samples,
                bounds,
            )
            if prepared is not None:
                self._minimize_item_mc(
                    model, item_idx, current_params, bounds, prepared, analytic=True
                )
                return

        valid_responses = item_responses[valid_mask]
        valid_theta = theta_samples[valid_mask]
        valid_weights = weights[valid_mask]

        def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
            self._set_item_params(model, item_idx, params)
            return -self._item_expected_log_likelihood(
                model,
                item_idx,
                valid_responses,
                valid_theta,
                valid_weights,
            )

        self._minimize_item_mc(
            model, item_idx, current_params, bounds, neg_expected_log_likelihood
        )

    def _uses_default_item_methods(self) -> bool:
        return type(self) in (
            MCEMEstimator,
            QMCEMEstimator,
            StochasticEMEstimator,
        ) and all(
            name not in vars(self) and getattr(type(self), name) is method
            for name, method in _DEFAULT_MC_ITEM_METHODS.items()
        )

    def _minimize_item_mc(
        self,
        model: BaseItemModel,
        item_idx: int,
        current_params: NDArray[np.float64],
        bounds: list[tuple[float, float]],
        objective: Callable[
            [NDArray[np.float64]], float | tuple[float, NDArray[np.float64]]
        ],
        *,
        analytic: bool = False,
    ) -> None:
        """Install a finite optimizer result, restoring the item after failures."""
        installed = False
        try:
            result = minimize(
                objective,
                x0=current_params,
                method="L-BFGS-B",
                jac=analytic,
                bounds=bounds,
                options={"maxiter": 50, "ftol": 1e-6},
            )
            candidate = np.asarray(result.x, dtype=np.float64)
            if (
                candidate.shape != current_params.shape
                or not np.all(np.isfinite(candidate))
                or not np.isfinite(result.fun)
            ):
                raise RuntimeError("item optimization returned invalid parameters")
            self._set_item_params(model, item_idx, candidate)
            installed = True
        finally:
            if not installed:
                self._set_item_params(model, item_idx, current_params)

    def _compute_standard_errors_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Optionally estimate diagonal complete-data curvature on the final draw."""
        if self.compute_standard_errors:
            from mirt.estimation._mc_information import mc_standard_errors

            return mc_standard_errors(
                self, model, responses, theta_samples, weights, self.se_step_size
            )
        standard_errors: dict[str, NDArray[np.float64]] = {}

        for name, values in model.parameters.items():
            if name == "discrimination" and model.model_name == "1PL":
                standard_errors[name] = np.zeros_like(values)
                continue

            se = np.full_like(values, np.nan)
            standard_errors[name] = se

        return standard_errors

    @staticmethod
    def _uses_default_information_model(model: BaseItemModel) -> bool:
        """Keep changed public model curves and parameter hooks authoritative."""
        return uses_builtin_model_hooks(model)


class QMCEMEstimator(MCEMEstimator):
    """Quasi-Monte Carlo EM estimator for IRT models.

    Uses low-discrepancy sequences (Sobol, Halton) instead of pseudo-random
    numbers for more uniform coverage of the integration space. This typically
    leads to faster convergence than standard MCEM.

    Parameters
    ----------
    n_samples : int
        Number of QMC samples per person per iteration.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print progress.
    seed : int or None
        Random seed for scrambling.
    sequence : str
        Type of low-discrepancy sequence: "sobol" or "halton".
    compute_standard_errors : bool
        Estimate approximate diagonal complete-data standard errors. Default False.
    se_step_size : float
        Positive finite differentiation step for standard errors. Default 1e-5.

    Notes
    -----
    QMCEM typically requires fewer samples than MCEM for the same accuracy
    because the quasi-random points fill the space more uniformly.
    Its shared ability grid also allows the M-step to aggregate expected
    response counts once, instead of evaluating each respondent's samples
    during every optimizer trial. Built-in logistic, affine, and polytomous
    models use analytic gradients.

    References
    ----------
    Niederreiter, H. (1992). Random number generation and quasi-Monte Carlo
        methods. Society for Industrial and Applied Mathematics.
    """

    _adaptive_sample_size = False

    def __init__(
        self,
        n_samples: int = 256,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        seed: int | None = None,
        sequence: Literal["sobol", "halton"] = "sobol",
        compute_standard_errors: bool = False,
        se_step_size: float = 1e-5,
    ) -> None:
        super().__init__(
            n_samples=n_samples,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            seed=seed,
            importance_sampling=True,
            compute_standard_errors=compute_standard_errors,
            se_step_size=se_step_size,
        )

        if sequence not in ("sobol", "halton"):
            raise ValueError("sequence must be 'sobol' or 'halton'")

        self.sequence = sequence

    def _e_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        L: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """E-step using Quasi-Monte Carlo sampling."""
        samples, weights, _ = self._e_step_mc_state(
            model, responses, prior_mean, L, n_factors
        )
        return samples, weights

    def _e_step_mc_state(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        L: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return shared QMC draws, weights, and their importance normalizers."""
        n_persons = responses.shape[0]

        if self.sequence == "sobol":
            sampler = qmc.Sobol(d=n_factors, scramble=True, seed=self.seed)
            exponent = int(np.ceil(np.log2(self.n_samples)))
            uniform_samples = sampler.random_base2(exponent)[: self.n_samples]
        else:
            sampler = qmc.Halton(d=n_factors, scramble=True, seed=self.seed)
            uniform_samples = sampler.random(self.n_samples)

        from scipy.stats import norm

        lower = np.nextafter(0.0, 1.0)
        upper = np.nextafter(1.0, 0.0)
        uniform_samples = np.clip(uniform_samples, lower, upper)
        z_base = norm.ppf(uniform_samples)

        theta_base = prior_mean + z_base @ L.T
        theta_samples = np.broadcast_to(
            theta_base[None, :, :],
            (n_persons, self.n_samples, n_factors),
        )

        if hasattr(model, "log_likelihood_batch"):
            log_likes = self._validated_log_likelihoods(
                model.log_likelihood_batch(responses, theta_base),
                (n_persons, self.n_samples),
            )
        else:
            log_likes = self._sample_log_likelihoods(model, responses, theta_samples)
        weights, log_normalizer = self._normalized_importance_weights(log_likes)

        return theta_samples, weights, log_normalizer

    def _sample_log_likelihoods(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Refresh shared-grid likelihoods without respondent/sample expansion."""
        if MCEMEstimator._sample_log_likelihoods is not _DEFAULT_MC_SAMPLE_LIKELIHOODS:
            return super()._sample_log_likelihoods(model, responses, theta_samples)
        n_persons = len(responses)
        samples = np.asarray(theta_samples)
        expected = (n_persons, self.n_samples, model.n_factors)
        if samples.shape != expected:
            raise ValueError(f"theta_samples must have shape {expected}")
        if (
            n_persons == 0
            or samples.strides[0] != 0
            or not hasattr(model, "log_likelihood_batch")
        ):
            return super()._sample_log_likelihoods(model, responses, theta_samples)
        grid = np.asarray(samples[0], dtype=np.float64)
        if not np.all(np.isfinite(grid)):
            raise ValueError("theta_samples must contain only finite values")
        values = self._validated_log_likelihoods(
            model.log_likelihood_batch(responses, grid), (n_persons, self.n_samples)
        )
        # The MC sampler owns its output and may update accepted cells in place.
        return np.array(values, dtype=np.float64, copy=True)

    def _m_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Optimize shared-grid items from expected counts without expanding theta."""
        from mirt.estimation._em_context import EMFitContext

        n_persons = responses.shape[0]
        if theta_samples.shape != (n_persons, self.n_samples, model.n_factors):
            raise ValueError("theta_samples has an incompatible shape")
        if weights.shape != (n_persons, self.n_samples):
            raise ValueError("weights has an incompatible shape")
        if theta_samples.strides[0] != 0 or not self._uses_default_item_methods():
            # Independent draws and custom item callbacks use the MC item path.
            super()._m_step_mc(model, responses, theta_samples, weights)
            return

        row_block = max(1, _MAX_QMC_COUNT_ELEMENTS // self.n_samples)
        for start in range(0, n_persons, row_block):
            block = weights[start : start + row_block]
            if not np.all(np.isfinite(block)) or np.any(block < 0.0):
                raise ValueError("weights must be finite and non-negative")

        if not any(np.any(mask) for mask in model.free_parameter_masks.values()):
            return

        theta = theta_samples[0]
        context = EMFitContext(responses)
        if model.is_polytomous:
            correct = observed = None
        else:
            correct, observed = context.expected_counts(weights)

        for item_idx in range(model.n_items):
            item_responses = responses[:, item_idx]
            if not np.any(item_responses >= 0):
                continue
            params, bounds = self._get_item_params_and_bounds(model, item_idx)
            if params.size == 0:
                continue

            objective: Callable[
                [NDArray[np.float64]], float | tuple[float, NDArray[np.float64]]
            ]
            analytic = False
            if model.is_polytomous:
                from mirt.estimation._polytomous_objective import (
                    prepare_polytomous_objective,
                )

                n_categories = model.n_categories[item_idx]
                if np.any(item_responses >= n_categories):
                    raise ValueError(
                        "model returned invalid item category probabilities"
                    )
                counts = context.expected_category_counts(
                    item_idx, n_categories, weights
                )
                prepared = prepare_polytomous_objective(
                    model, item_idx, theta, counts, PROB_EPSILON, max_probability=1.0
                )
                analytic = prepared is not None
                if prepared is not None:
                    objective = prepared
                else:

                    def objective(trial):
                        self._set_item_params(model, item_idx, trial)
                        probabilities = np.asarray(
                            model.probability(theta, item_idx), dtype=np.float64
                        )
                        if probabilities.shape != counts.shape or not np.all(
                            np.isfinite(probabilities)
                        ):
                            raise ValueError(
                                "model returned invalid item category probabilities"
                            )
                        return -float(
                            np.sum(
                                xlogy(counts, np.clip(probabilities, PROB_EPSILON, 1.0))
                            )
                        )

            else:
                from mirt.estimation._affine_objective import prepare_affine_objective
                from mirt.estimation._dichotomous_objective import (
                    prepare_dichotomous_objective,
                )

                n_k, r_k = observed[item_idx], correct[item_idx]
                prepared = prepare_dichotomous_objective(
                    model, item_idx, theta, n_k, r_k, PROB_EPSILON, bounds
                )
                if prepared is None:
                    prepared = prepare_affine_objective(
                        model, item_idx, theta, n_k, r_k, PROB_EPSILON, bounds
                    )
                analytic = prepared is not None
                if prepared is not None:
                    objective = prepared
                else:

                    def objective(trial):
                        self._set_item_params(model, item_idx, trial)
                        probabilities = np.asarray(
                            model.probability(theta, item_idx), dtype=np.float64
                        ).reshape(-1)
                        if probabilities.shape != (self.n_samples,) or not np.all(
                            np.isfinite(probabilities)
                        ):
                            raise ValueError(
                                "model returned invalid item probabilities"
                            )
                        probabilities = np.clip(
                            probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON
                        )
                        return -float(
                            np.sum(
                                xlogy(r_k, probabilities)
                                + xlog1py(n_k - r_k, -probabilities)
                            )
                        )

            self._minimize_item_mc(
                model, item_idx, params, bounds, objective, analytic=analytic
            )


class StochasticEMEstimator(MCEMEstimator):
    """Stochastic EM (SEM) estimator for IRT models.

    SEM draws a single sample from the posterior in the E-step instead
    of computing expectations. This makes each iteration faster but
    noisier, requiring more iterations to converge.

    Parameters
    ----------
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print progress.
    seed : int or None
        Random seed.
    n_chains : int
        Number of independent chains to average over.
    compute_standard_errors : bool
        Estimate approximate diagonal complete-data standard errors. Default False.
    se_step_size : float
        Positive finite differentiation step for standard errors. Default 1e-5.

    Notes
    -----
    SEM can be useful for very large datasets where computing full
    expectations is too expensive. It converges to a neighborhood of
    the MLE rather than exactly to it.

    References
    ----------
    Celeux, G., & Diebolt, J. (1985). The SEM algorithm: a probabilistic
        teacher algorithm derived from the EM algorithm for the mixture
        problem. Computational Statistics Quarterly, 2(1), 73-82.
    """

    _minimum_samples = 1
    _adaptive_sample_size = False

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-4,
        verbose: bool = False,
        seed: int | None = None,
        n_chains: int = 5,
        compute_standard_errors: bool = False,
        se_step_size: float = 1e-5,
    ) -> None:
        super().__init__(
            n_samples=n_chains,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            seed=seed,
            importance_sampling=False,
            compute_standard_errors=compute_standard_errors,
            se_step_size=se_step_size,
        )
        self.n_chains = self.n_samples

    def _e_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        L: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """E-step: sample from posterior using Metropolis-Hastings."""
        return self._draw_posterior_samples(
            model,
            responses,
            prior_mean,
            L,
            n_factors,
        )


_DEFAULT_MC_ITEM_METHODS = {
    name: getattr(MCEMEstimator, name)
    for name in (
        "_item_expected_log_likelihood",
        "_get_item_params_and_bounds",
        "_set_item_params",
        "_optimize_item_mc",
    )
}

_DEFAULT_MC_SAMPLE_LIKELIHOODS = MCEMEstimator._sample_log_likelihoods

_DEFAULT_QMC_BATCH_LIKELIHOODS = {
    original_model_hook(cls, "log_likelihood_batch")
    for cls in (
        DichotomousItemModel,
        PolytomousItemModel,
        GradedResponseModel,
        GeneralizedPartialCredit,
    )
}

_DEFAULT_MC_STATE_METHODS = {
    cls: {
        name: getattr(cls, name)
        for name in (
            "_e_step_mc",
            "_e_step_mc_state",
            "_draw_posterior_samples",
            "_draw_posterior_state",
            "_gaussian_log_kernel",
            "_sample_log_likelihoods",
            "_normalized_importance_weights",
            "_validated_log_likelihoods",
            "_estimate_marginal_ll",
            "_refresh_mc_state",
        )
    }
    for cls in (MCEMEstimator, QMCEMEstimator, StochasticEMEstimator)
}
