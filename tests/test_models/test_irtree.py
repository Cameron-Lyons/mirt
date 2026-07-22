"""Smoke tests for IRTree models."""

import numpy as np

from mirt.models.irtree import IRTreeModel, IRTreeSpec


def test_irtree_bockenholt_category_probs_sum_to_one():
    spec = IRTreeSpec.bockenholt_adi()
    model = IRTreeModel(n_items=3, tree_spec=spec)
    theta = np.zeros((5, model.n_traits))

    probs = model.probability(theta)

    assert probs.shape == (5, 3, 5)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-6)
    assert np.all((probs >= 0) & (probs <= 1))
