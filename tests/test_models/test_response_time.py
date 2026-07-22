"""Smoke tests for response time models."""

import numpy as np

from mirt.models.response_time import ResponseTimeModel


def test_response_time_accuracy_probability():
    model = ResponseTimeModel(n_items=5)
    theta = np.linspace(-2, 2, 8)
    probs = model.accuracy_probability(theta)

    assert probs.shape == (8, 5)
    assert np.all((probs > 0) & (probs < 1))


def test_response_time_simulate_and_joint_ll():
    model = ResponseTimeModel(n_items=4)
    responses, rts, theta, tau = model.simulate(n_persons=10, seed=0)

    assert responses.shape == (10, 4)
    assert rts.shape == (10, 4)
    assert theta.shape == (10,)
    assert tau.shape == (10,)
    assert np.all((responses == 0) | (responses == 1))
    assert np.all(rts > 0)

    log_rt = np.log(rts)
    ll = model.joint_log_likelihood(responses, log_rt, theta, tau)
    assert ll.shape == (10,)
    assert np.all(np.isfinite(ll))
