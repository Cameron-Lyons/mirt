"""Nested invariance fits of an exploratory two-factor multigroup model."""

import numpy as np

from mirt.models.dichotomous import TwoParameterLogistic
from mirt.multigroup import MultigroupEMEstimator, MultigroupModel


def _group_responses(seed=0, n_persons=500, n_items=10):
    rng = np.random.default_rng(seed)
    half = n_items // 2
    slopes = np.zeros((n_items, 2))
    slopes[:half, 0] = rng.uniform(0.8, 1.6, half)
    slopes[half:, 1] = rng.uniform(0.8, 1.6, n_items - half)
    intercepts = rng.normal(0.0, 1.0, n_items)
    responses = []
    for mean in ([0.0, 0.0], [-0.4, 0.3]):
        theta = rng.multivariate_normal(mean, [[1.0, 0.4], [0.4, 1.0]], n_persons)
        probability = 1.0 / (1.0 + np.exp(-(theta @ slopes.T + intercepts)))
        responses.append((rng.random(probability.shape) < probability).astype(int))
    return responses, n_items


def _fit(responses, n_items, invariance):
    model = MultigroupModel(TwoParameterLogistic(n_items=n_items, n_factors=2), 2)
    estimator = MultigroupEMEstimator(n_quadpts=11, max_iter=300)
    return estimator.fit(model, responses, invariance=invariance)


def test_configural_two_factor_fit_nests_the_metric_fit():
    responses, n_items = _group_responses()
    configural = _fit(responses, n_items, "configural")
    metric = _fit(responses, n_items, "metric")

    # Equal default slopes kept the configural factors identical, so its
    # log-likelihood fell below that of the nested metric model.
    assert configural.log_likelihood >= metric.log_likelihood - 1e-3
    for group in range(2):
        slopes = configural.model.get_group_model(group).parameters["discrimination"]
        assert np.max(np.abs(slopes[:, 0] - slopes[:, 1])) > 0.1
