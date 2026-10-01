"""Native adaptive summaries must rescore their own response paths correctly."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt import is_rust_available
from mirt.cat import CATEngine
from mirt.models import TwoParameterLogistic
from mirt.scoring import fscores

pytestmark = pytest.mark.skipif(
    not is_rust_available(), reason="native extension unavailable"
)


@pytest.mark.parametrize("slope", [1.5, 100.0])
@pytest.mark.parametrize("n_quadpts", [6, 11])
def test_native_results_match_rescoring_actual_paths(slope, n_quadpts):
    model = TwoParameterLogistic(n_items=4)
    model.set_parameters(discrimination=np.full(4, slope), difficulty=np.zeros(4))
    model._is_fitted = True
    engine = CATEngine(
        model, n_quadpts=n_quadpts, max_items=4, min_items=4, se_threshold=1e-8, seed=8
    )
    results = engine.run_batch_simulation([0.0], n_replications=12, use_rust=True)
    responses = np.full((len(results), 4), -1)
    for row, result in enumerate(results):
        responses[row, result.items_administered] = result.responses
    assert any(0 < result.responses.sum() < 4 for result in results)
    scored = fscores(model, responses, n_quadpts=n_quadpts)
    assert_allclose([result.theta for result in results], scored.theta, atol=1e-13)
    assert_allclose(
        [result.standard_error for result in results], scored.standard_error, atol=1e-13
    )


def test_native_conditional_mse_uses_same_bounded_response_scores():
    model = TwoParameterLogistic(n_items=4)
    model.set_parameters(discrimination=np.full(4, 100.0), difficulty=np.zeros(4))
    model._is_fitted = True
    engine = CATEngine(
        model, n_quadpts=6, max_items=4, min_items=4, se_threshold=1e-8, seed=8
    )
    # At the first theta, batch and MSE use the same per-replication seeds.
    results = engine.run_batch_simulation([0.0], n_replications=12, use_rust=True)
    responses = np.full((len(results), 4), -1)
    for row, result in enumerate(results):
        responses[row, result.items_administered] = result.responses
    scored = fscores(model, responses, n_quadpts=6)
    thetas, bias, mse, average_items = engine.compute_conditional_mse(
        [0.0], n_replications=12, use_rust=True
    )
    assert_allclose(thetas, [0.0])
    assert_allclose(bias, [scored.theta.mean()], atol=1e-13)
    assert_allclose(mse, [(scored.theta**2).mean()], atol=1e-13)
    assert_allclose(average_items, [4.0])
