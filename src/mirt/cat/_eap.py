"""Bounded EAP scoring for independent built-in binary adaptive tests."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from mirt._model_defaults import original_model_hook, uses_builtin_model_hooks
from mirt.constants import PROB_EPSILON

_MAX_CURVE_VALUES = 131_072


def score_binary_eap(
    engine: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Score only administered curves; preserve customized likelihood paths.

    Cache the parameter-independent quadrature grid, but reevaluate response
    history against the current model every time. Parameter and structural
    changes between responses therefore cannot leave stale posterior evidence.
    """
    from mirt.models.base import DichotomousItemModel
    from mirt.scoring._common import build_quadrature

    model = engine.model
    if model.is_polytomous or not uses_builtin_model_hooks(model, likelihood=True):
        return None
    for name in ("log_likelihood", "log_likelihood_batch"):
        if original_model_hook(type(model), name) is not original_model_hook(
            DichotomousItemModel, name
        ):
            return None
    if not model.is_fitted:
        raise ValueError("Model must be fitted before scoring")

    # Validate before comparing cache keys: 21.0 equals 21 but is not a valid
    # quadrature count. Reuse must preserve the public scorer's input contract.
    from mirt.scoring.eap import EAPScorer

    EAPScorer(n_quadpts=engine.n_quadpts)
    key = (engine.n_quadpts, model.n_factors)
    cached = getattr(engine, "_eap_quadrature", None)
    if cached is None or cached[0] != key:
        points, weights = build_quadrature(
            n_quadpts=engine.n_quadpts,
            n_factors=model.n_factors,
            prior_mean=None,
            prior_cov=None,
        )
        cached = (key, points, np.log(weights + 1e-300))
        engine._eap_quadrature = cached
    _, points, log_weights = cached
    log_posterior = log_weights.copy()

    for item_idx, response in zip(
        engine._items_administered, engine._responses, strict=True
    ):
        for start in range(0, len(points), _MAX_CURVE_VALUES):
            stop = min(start + _MAX_CURVE_VALUES, len(points))
            probabilities = np.clip(
                model.probability(points[start:stop], item_idx=item_idx),
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            ).reshape(-1)
            if response == 1:
                np.log(probabilities, out=probabilities)
            else:
                np.negative(probabilities, out=probabilities)
                np.log1p(probabilities, out=probabilities)
            log_posterior[start:stop] += probabilities

    normalizer = logsumexp(log_posterior)
    if not np.isfinite(normalizer):
        raise ValueError("model likelihoods must produce finite posterior mass")
    log_posterior -= normalizer
    np.exp(log_posterior, out=log_posterior)
    theta = log_posterior @ points
    covariance = np.zeros((model.n_factors, model.n_factors))
    # Keep moment scratch bounded even when a multidimensional grid is large.
    chunk_size = max(1, _MAX_CURVE_VALUES // model.n_factors)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        centered = points[start:stop] - theta
        covariance += centered.T @ (log_posterior[start:stop, None] * centered)
    return theta, covariance
