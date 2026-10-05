from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.results.ability_posterior import AbilityPosteriorResult
from mirt.results.score_result import ScoreResult
from mirt.scoring._common import (
    build_quadrature,
    unique_response_patterns,
    validate_scoring_responses,
)
from mirt.utils.numeric import logsumexp_axis1

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult
    from mirt.scoring._bifactor import ReducedBifactorGrid


_TARGET_WORKING_BYTES = 32 * 1024 * 1024
_PATTERN_SAMPLE_SIZE = 1_024
# Require a projected fourfold reduction so sorting also pays off for the
# fastest native likelihood implementations.
_MAX_SAMPLE_UNIQUE_FRACTION = 0.25
# EAP integrates over a full tensor-product grid, so its size grows as
# n_quadpts ** n_factors. These per-dimension defaults keep three or more
# factors tractable; one and two factors keep the historical 49 points.
_DEFAULT_QUADPTS_BY_FACTORS = {1: 49, 2: 49, 3: 21, 4: 9, 5: 7}
_DEFAULT_QUADPTS_HIGH_DIMENSIONAL = 5
# Larger grids still run, with a warning; 21 points in five dimensions is the
# largest grid the package requests internally.
_LARGE_GRID_NODES = 21**5
# Bifactor EAP integrates two-dimensional (general by specific) grids, so it
# keeps the two-factor default whatever the number of specific factors.
_DEFAULT_BIFACTOR_QUADPTS = _DEFAULT_QUADPTS_BY_FACTORS[2]
# Automatic product grids coarser than this per dimension visibly bias the
# posterior summaries of bifactor models (by up to 0.4 at five points).
_MIN_ACCURATE_BIFACTOR_QUADPTS = 11


def _default_n_quadpts(n_factors: int) -> int:
    """Return the default EAP quadrature points per latent dimension.

    The defaults are 49 points for one or two factors, 21 for three, 9 for
    four, 7 for five, and 5 for six or more.
    """
    return _DEFAULT_QUADPTS_BY_FACTORS.get(
        int(n_factors), _DEFAULT_QUADPTS_HIGH_DIMENSIONAL
    )


def _eap_response_patterns(
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.int_], NDArray[np.intp]]:
    """Compress rows when a bounded sample predicts useful likelihood reuse."""
    n_persons = responses.shape[0]
    if n_persons <= _PATTERN_SAMPLE_SIZE:
        return unique_response_patterns(responses)

    sample_indices = np.linspace(
        0,
        n_persons - 1,
        _PATTERN_SAMPLE_SIZE,
        dtype=np.intp,
    )
    sample_patterns, _ = unique_response_patterns(responses[sample_indices])
    if sample_patterns.shape[0] > (_MAX_SAMPLE_UNIQUE_FRACTION * _PATTERN_SAMPLE_SIZE):
        return responses, np.arange(n_persons, dtype=np.intp)
    return unique_response_patterns(responses)


class EAPScorer:
    """Expected a posteriori scoring on a Gauss-Hermite quadrature grid.

    Parameters
    ----------
    n_quadpts : int, optional
        Quadrature points per latent dimension. ``None`` chooses a size from
        the model's factor count when scoring: 49 points for one or two
        factors, 21 for three, 9 for four, 7 for five, and 5 for six or more.
        Bifactor models that :meth:`score` integrates by dimension reduction
        (see Notes) default to 49 points whatever their factor count.
    prior_mean : ndarray, optional
        Prior mean for theta. Default zeros.
    prior_cov : ndarray, optional
        Prior covariance for theta. Default identity.
    batch_size : int, optional
        Maximum response rows per likelihood batch.

    Notes
    -----
    EAP integrates over a tensor-product grid of ``n_quadpts ** n_factors``
    nodes. For a :class:`~mirt.models.BifactorModel` whose prior makes the
    specific factors conditionally independent given the general factor (any
    diagonal covariance, for example), :meth:`score` instead integrates each
    specific factor jointly with the general factor on its own
    ``n_quadpts ** 2`` grid (Gibbons & Hedeker, 1992). This reproduces the
    product-grid estimates at the same ``n_quadpts`` to rounding error, at a
    cost linear rather than exponential in the number of specific factors.
    Bifactor models scored on an automatic product grid coarser than 11 points
    per dimension emit a ``RuntimeWarning``.

    References
    ----------
    Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
        analysis. Psychometrika, 57(3), 423-436.
    """

    def __init__(
        self,
        n_quadpts: int | None = None,
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
        batch_size: int | None = None,
    ) -> None:
        if n_quadpts is not None and (
            isinstance(n_quadpts, (bool, np.bool_))
            or not isinstance(n_quadpts, (int, np.integer))
            or n_quadpts < 5
        ):
            raise ValueError("n_quadpts should be at least 5")

        self.n_quadpts = None if n_quadpts is None else int(n_quadpts)
        self.prior_mean = (
            None
            if prior_mean is None
            else np.array(prior_mean, dtype=np.float64, copy=True)
        )
        self.prior_cov = (
            None
            if prior_cov is None
            else np.array(prior_cov, dtype=np.float64, copy=True)
        )
        if batch_size is not None and (
            isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer or None")
        self.batch_size = None if batch_size is None else int(batch_size)

    def _quadrature(
        self,
        model: BaseItemModel,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build the product grid, resolving the automatic grid size."""
        from mirt.models.bifactor import BifactorModel

        n_factors = model.n_factors
        default = _default_n_quadpts(n_factors)
        n_quadpts = default if self.n_quadpts is None else self.n_quadpts
        n_nodes = n_quadpts**n_factors
        if n_nodes > _LARGE_GRID_NODES:
            advice = (
                f"omit n_quadpts to use {default} points per dimension"
                if n_quadpts > default
                else "consider MAP scoring for this many factors"
            )
            warnings.warn(
                f"EAP quadrature with n_quadpts={n_quadpts} and {n_factors} "
                f"factors has {n_nodes} grid nodes, which is slow and "
                f"memory-intensive; {advice}",
                RuntimeWarning,
                stacklevel=3,
            )
        if (
            self.n_quadpts is None
            and n_quadpts < _MIN_ACCURATE_BIFACTOR_QUADPTS
            and isinstance(model, BifactorModel)
        ):
            advice = (
                "; fscores(method='EAP') integrates this model by dimension "
                "reduction instead"
                if self._bifactor_grid(model) is not None
                else ""
            )
            warnings.warn(
                f"EAP for this {n_factors}-factor bifactor model uses a product "
                f"grid of only {n_quadpts} points per dimension, which biases "
                f"posterior summaries; pass n_quadpts to choose a finer grid{advice}",
                RuntimeWarning,
                stacklevel=3,
            )
        return build_quadrature(
            n_quadpts=n_quadpts,
            n_factors=n_factors,
            prior_mean=self.prior_mean,
            prior_cov=self.prior_cov,
        )

    def _bifactor_grid(self, model: BaseItemModel) -> ReducedBifactorGrid | None:
        """Return the dimension-reduced bifactor grid when it applies."""
        from mirt.models.bifactor import BifactorModel

        if not isinstance(model, BifactorModel):
            return None
        from mirt.scoring._bifactor import reduced_bifactor_grid

        n_quadpts = (
            _DEFAULT_BIFACTOR_QUADPTS if self.n_quadpts is None else self.n_quadpts
        )
        return reduced_bifactor_grid(model, n_quadpts, self.prior_mean, self.prior_cov)

    def _resolve_batch_size(
        self,
        *,
        n_patterns: int,
        n_items: int,
        n_quad: int,
    ) -> int:
        """Choose a pattern batch that bounds temporary likelihood storage."""
        if self.batch_size is not None:
            return min(self.batch_size, n_patterns)

        # Generic likelihood evaluation uses boolean and float response matrices,
        # while posterior normalization holds several theta-grid matrices. This
        # conservative estimate keeps their combined working set near 32 MiB.
        bytes_per_pattern = 17 * n_items + 32 * n_quad
        automatic_size = max(1, _TARGET_WORKING_BYTES // bytes_per_pattern)
        return min(automatic_size, n_patterns)

    @staticmethod
    def _posterior_batch(
        model: BaseItemModel,
        responses: NDArray[np.int_],
        quad_points: NDArray[np.float64],
        log_weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalize one likelihood batch and return its log marginal values."""
        posterior = np.array(
            model.log_likelihood_batch(responses, quad_points),
            dtype=np.float64,
            copy=True,
        )
        expected_shape = (responses.shape[0], quad_points.shape[0])
        if posterior.shape != expected_shape:
            raise ValueError(
                f"model log-likelihood batch has shape {posterior.shape}, "
                f"expected {expected_shape}"
            )

        posterior += log_weights[None, :]
        log_marginal = logsumexp_axis1(posterior)
        if not np.all(np.isfinite(log_marginal)):
            raise ValueError("model likelihoods must produce finite posterior mass")
        posterior -= log_marginal[:, None]
        np.exp(posterior, out=posterior)
        return posterior, log_marginal

    def _product_grid_moments(
        self,
        model: BaseItemModel,
        patterns: NDArray[np.int_],
        quad_points: NDArray[np.float64],
        quad_weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return posterior means and SDs of patterns on the product grid."""
        center = quad_weights @ quad_points
        centered_points = quad_points - center
        centered_points_squared = centered_points**2
        log_weights = np.log(quad_weights + 1e-300)
        n_patterns = patterns.shape[0]
        pattern_theta = np.empty((n_patterns, model.n_factors), dtype=np.float64)
        pattern_se = np.empty_like(pattern_theta)
        batch_size = self._resolve_batch_size(
            n_patterns=n_patterns,
            n_items=model.n_items,
            n_quad=quad_points.shape[0],
        )

        for start in range(0, n_patterns, batch_size):
            stop = min(start + batch_size, n_patterns)
            posterior, _ = self._posterior_batch(
                model,
                patterns[start:stop],
                quad_points,
                log_weights,
            )

            centered_mean = posterior @ centered_points
            pattern_theta[start:stop] = centered_mean + center
            variance = posterior @ centered_points_squared - centered_mean**2
            np.maximum(variance, 0.0, out=variance)
            np.sqrt(variance, out=pattern_se[start:stop])
        return pattern_theta, pattern_se

    def score(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> ScoreResult:
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        responses = validate_scoring_responses(model, responses)
        n_factors = model.n_factors

        bifactor_grid = self._bifactor_grid(model)
        quadrature = None if bifactor_grid is not None else self._quadrature(model)
        if responses.shape[0] == 0:
            shape = (0,) if n_factors == 1 else (0, n_factors)
            return ScoreResult(
                theta=np.empty(shape, dtype=np.float64),
                standard_error=np.empty(shape, dtype=np.float64),
                method="EAP",
            )

        patterns, inverse = _eap_response_patterns(responses)
        if bifactor_grid is not None:
            from mirt.scoring._bifactor import bifactor_eap

            pattern_theta, pattern_se = bifactor_eap(
                bifactor_grid,
                patterns,
                lambda n_nodes: self._resolve_batch_size(
                    n_patterns=patterns.shape[0],
                    n_items=model.n_items,
                    n_quad=n_nodes,
                ),
            )
        else:
            assert quadrature is not None
            pattern_theta, pattern_se = self._product_grid_moments(
                model, patterns, *quadrature
            )

        theta_eap = pattern_theta[inverse]
        theta_se = pattern_se[inverse]

        if n_factors == 1:
            theta_eap = theta_eap.ravel()
            theta_se = theta_se.ravel()

        return ScoreResult(
            theta=theta_eap,
            standard_error=theta_se,
            method="EAP",
        )

    def posterior(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        *,
        person_ids: list[Any] | NDArray[Any] | None = None,
    ) -> AbilityPosteriorResult:
        """Return normalized ability distributions on the EAP quadrature grid.

        Repeated response patterns reuse likelihood evaluations. This method
        retains one probability per respondent and grid point; callers should
        account for the returned ``n_persons * n_points`` array when choosing
        ``n_quadpts`` for multidimensional models.

        The joint posterior always uses the full product grid, also for
        bifactor models, which :meth:`score` integrates by dimension
        reduction. An automatic bifactor grid coarser than 11 points per
        dimension emits a ``RuntimeWarning``, since it biases the posterior
        summaries.
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        responses = validate_scoring_responses(model, responses)
        person_ids = AbilityPosteriorResult._validated_person_ids(
            person_ids, responses.shape[0]
        )
        quad_points, quad_weights = self._quadrature(model)
        n_persons = responses.shape[0]
        posterior_weights = np.empty(
            (n_persons, quad_points.shape[0]),
            dtype=np.float64,
        )
        log_marginal = np.empty(n_persons, dtype=np.float64)
        if n_persons:
            patterns, inverse = _eap_response_patterns(responses)
            n_patterns = patterns.shape[0]
            log_weights = np.log(quad_weights + 1e-300)
            batch_size = self._resolve_batch_size(
                n_patterns=n_persons,
                n_items=model.n_items,
                n_quad=quad_points.shape[0],
            )
            if n_patterns < n_persons:
                # Group respondent indices once. Expansion is also batched so a
                # common pattern cannot allocate another full posterior matrix.
                grouped_rows = np.argsort(inverse, kind="stable")
                offsets = np.concatenate(
                    ([0], np.cumsum(np.bincount(inverse, minlength=n_patterns)))
                )
            for start in range(0, n_patterns, batch_size):
                stop = min(start + batch_size, n_patterns)
                batch_weights, batch_log_marginal = self._posterior_batch(
                    model,
                    patterns[start:stop],
                    quad_points,
                    log_weights,
                )
                if n_patterns == n_persons:
                    posterior_weights[start:stop] = batch_weights
                    log_marginal[start:stop] = batch_log_marginal
                else:
                    for row_start in range(offsets[start], offsets[stop], batch_size):
                        rows = grouped_rows[
                            row_start : min(row_start + batch_size, offsets[stop])
                        ]
                        pattern_indices = inverse[rows] - start
                        posterior_weights[rows] = batch_weights[pattern_indices]
                        log_marginal[rows] = batch_log_marginal[pattern_indices]

        return AbilityPosteriorResult._from_owned_arrays(
            points=quad_points,
            weights=posterior_weights,
            log_marginal_likelihood=log_marginal,
            person_ids=person_ids,
        )

    def __repr__(self) -> str:
        if self.batch_size is None:
            return f"EAPScorer(n_quadpts={self.n_quadpts})"
        return f"EAPScorer(n_quadpts={self.n_quadpts}, batch_size={self.batch_size})"


def ability_posterior(
    model_or_result: BaseItemModel | FitResult,
    responses: ArrayLike,
    *,
    n_quadpts: int | None = None,
    prior_mean: NDArray[np.float64] | None = None,
    prior_cov: NDArray[np.float64] | None = None,
    batch_size: int | None = None,
    person_ids: list[Any] | NDArray[Any] | None = None,
) -> AbilityPosteriorResult:
    """Compute normalized posterior ability distributions for respondents.

    ``responses`` may be an array or a pandas or polars DataFrame; negative
    codes, ``NaN`` and the nulls of nullable DataFrame columns denote missing
    responses, as in :func:`mirt.fit_mirt`. The posterior always lives on the
    full product grid. Coarse grids bias posterior summaries, so for a
    bifactor model an automatic grid coarser than 11 points per dimension
    warns; :func:`mirt.fscores` scores bifactor models by dimension reduction
    on a fine grid instead.

    ``model_or_result`` may be either a fitted item model or the ``FitResult``
    returned by :func:`mirt.fit_mirt`. ``n_quadpts`` is the number of grid
    points per latent dimension; by default it is 49 for one or two factors,
    21 for three, 9 for four, 7 for five, and 5 for six or more.
    ``prior_mean`` and ``prior_cov`` default to the ``latent_mean`` and
    ``latent_covariance`` of a ``FitResult`` when it has them, and to the
    standard normal otherwise.
    """
    from mirt.results._common import resolve_latent_prior

    model, prior_mean, prior_cov = resolve_latent_prior(
        model_or_result, prior_mean, prior_cov
    )
    scorer = EAPScorer(
        n_quadpts=n_quadpts,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        batch_size=batch_size,
    )
    return scorer.posterior(model, responses, person_ids=person_ids)
