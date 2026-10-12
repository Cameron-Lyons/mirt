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
    options = dict(n_quadpts=6, max_items=4, min_items=4, se_threshold=1e-8)
    engine = CATEngine(model, **options, seed=8)
    # Saturated curves make each session's score a function of its number of
    # correct answers, so the MSE is a mixture of the bounded fscores values.
    pattern_scores = []
    for n_correct in range(5):
        responses = np.zeros((1, 4), dtype=int)
        responses[0, :n_correct] = 1
        pattern_scores.append(fscores(model, responses, n_quadpts=6).theta[0])
    native = engine.compute_conditional_mse([0.0], n_replications=4000, use_rust=True)
    reference = CATEngine(model, **options, seed=9).compute_conditional_mse(
        [0.0], n_replications=4000, use_rust=False
    )
    thetas, bias, mse, average_items = native
    assert_allclose(thetas, [0.0])
    assert_allclose(average_items, [4.0])
    # Correct answers are Binomial(4, 1/2) at theta = 0 on both paths.
    weights = np.array([1, 4, 6, 4, 1]) / 16
    expected_mse = weights @ np.square(pattern_scores)
    assert_allclose(mse, [expected_mse], rtol=0.05)
    assert_allclose(mse, reference[2], rtol=0.05)
    assert abs(bias[0]) < 0.05
