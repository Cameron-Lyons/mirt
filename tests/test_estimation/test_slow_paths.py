"""Slow / high-cost estimation paths for the weekly CI suite."""

from __future__ import annotations

import numpy as np
import pytest

import mirt


@pytest.mark.slow
def test_high_dimensional_em_fit() -> None:
    responses = mirt.simdata(
        model="2PL", n_persons=300, n_items=20, n_factors=2, seed=99
    )
    result = mirt.fit_mirt(
        responses,
        model="2PL",
        n_factors=2,
        n_quadpts=11,
        max_iter=40,
        tol=1e-3,
        use_rust=False,
    )
    assert np.isfinite(result.log_likelihood)


@pytest.mark.slow
def test_gibbs_sampler_longer_chain() -> None:
    responses = mirt.simdata(model="2PL", n_persons=80, n_items=8, seed=17)
    model = mirt.TwoParameterLogistic(n_items=8)
    sampler = mirt.GibbsSampler(n_iter=200, burnin=50, thin=2, seed=17)
    result = sampler.fit(model, responses)
    assert result.n_iterations >= 50
    assert np.isfinite(result.log_likelihood)
