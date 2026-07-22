"""Smoke tests for compensatory / noncompensatory models."""

import numpy as np

from mirt.models.compensatory import (
    DisjunctiveModel,
    NoncompensatoryModel,
    PartiallyCompensatoryModel,
)


def test_partially_compensatory_probability_shape():
    model = PartiallyCompensatoryModel(n_items=5, n_factors=2)
    theta = np.zeros((10, 2))
    probs = model.probability(theta)

    assert probs.shape == (10, 5)
    assert np.all((probs > 0) & (probs < 1))


def test_noncompensatory_probability_shape():
    model = NoncompensatoryModel(n_items=5, n_factors=2)
    theta = np.zeros((10, 2))
    probs = model.probability(theta)

    assert probs.shape == (10, 5)
    assert np.all((probs > 0) & (probs < 1))


def test_disjunctive_probability_shape():
    model = DisjunctiveModel(n_items=5, n_factors=2)
    theta = np.zeros((10, 2))
    probs = model.probability(theta)

    assert probs.shape == (10, 5)
    assert np.all((probs > 0) & (probs < 1))
