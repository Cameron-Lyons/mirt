"""Exact diagnostics must retain shared latent-pattern dependence."""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt import compute_fit_indices, compute_m2
from mirt.diagnostics.itemfit import compute_itemfit, compute_s_x2
from mirt.models.cdm_advanced import HigherOrderCDM


def test_shared_mastery_has_covariance_after_conditioning_on_higher_order_trait():
    """Enumerate a shared binary mastery state without diagnostic helpers.

    At theta=0 its two states have equal probability. Each default item has
    success probabilities .2 and .6 for nonmastery and mastery, respectively.
    The same state governs both items, giving covariance .04; multiplying the
    marginal .4 curves instead would incorrectly give covariance zero.
    """
    model = HigherOrderCDM(2, 1, np.ones((2, 1), dtype=int))
    theta = np.zeros((1, 1))
    patterns = np.array(list(product([0, 1], repeat=2)))
    shared_mass = np.array(
        [
            0.5 * np.prod(np.where(row == 1, 0.2, 0.8))
            + 0.5 * np.prod(np.where(row == 1, 0.6, 0.4))
            for row in patterns
        ]
    )
    assert_allclose(shared_mass, [0.4, 0.2, 0.2, 0.2])
    expected_scores = shared_mass @ patterns
    covariance = (patterns.T * shared_mass) @ patterns - np.outer(
        expected_scores, expected_scores
    )
    assert_allclose(covariance, [[0.24, 0.04], [0.04, 0.24]])
    assert_allclose(model.pattern_probability(theta), [[0.5, 0.5]])
    assert_allclose(model.probability(theta), expected_scores[None, :])
    marginal_product = np.prod(
        np.where(patterns == 1, expected_scores, 1 - expected_scores), axis=1
    )
    assert_allclose(marginal_product, [0.36, 0.24, 0.24, 0.16])
    # The model's existing likelihood evaluates a product approximation. Exact
    # limited-information and score-distribution diagnostics cannot inherit it
    # as the shared-attribute model's true conditional response distribution.
    assert_allclose(
        np.exp(model.log_likelihood(patterns, np.zeros((4, 1)))), marginal_product
    )
    assert not np.allclose(shared_mass, marginal_product)


@pytest.mark.parametrize("diagnostic", ["M2", "fit_indices", "S_X2", "itemfit"])
def test_joint_diagnostics_reject_shared_mastery_marginal_curves(diagnostic):
    model = HigherOrderCDM(2, 1, np.ones((2, 1), dtype=int))
    responses = np.tile(np.array(list(product([0, 1], repeat=2))), (10, 1))
    with pytest.raises(ValueError, match="shared mastery-pattern integration"):
        if diagnostic == "M2":
            compute_m2(model, responses)
        elif diagnostic == "fit_indices":
            compute_fit_indices(model, responses)
        elif diagnostic == "S_X2":
            compute_s_x2(model, responses, item_parameter_counts=[0, 0])
        else:
            compute_itemfit(
                model, responses, statistics=["S_X2"], item_parameter_counts=[0, 0]
            )


def test_shared_mastery_rejection_is_limited_to_joint_diagnostics():
    model = HigherOrderCDM(2, 1, np.ones((2, 1), dtype=int))
    responses = np.array(list(product([0, 1], repeat=2)))
    result = compute_itemfit(
        model, responses, theta=np.zeros((4, 1)), statistics=["infit", "outfit"]
    )
    assert np.all(np.isfinite(result["infit"]))
    assert np.all(np.isfinite(result["outfit"]))
