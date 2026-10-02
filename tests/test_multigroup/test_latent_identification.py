"""Affine IRT likelihood invariance and conventional latent identification."""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.models import (
    GradedResponseModel,
    OneParameterLogistic,
    PartialCreditModel,
    TwoParameterLogistic,
)
from mirt.multigroup._identification import infer_latent_identification
from mirt.multigroup.estimator import MultigroupEMEstimator
from mirt.multigroup.invariance import InvarianceSpec
from mirt.multigroup.model import MultigroupModel


def pattern_masses(model, mean, standard_deviation):
    """Independent marginal likelihood by Gaussian integration and enumeration."""
    roots, weights = np.polynomial.hermite.hermgauss(81)
    theta = (mean + standard_deviation * np.sqrt(2) * roots)[:, None]
    probabilities = model.probability(theta)
    masses = []
    for pattern in product(range(2), repeat=model.n_items):
        likelihood = np.ones(len(theta))
        for item, response in enumerate(pattern):
            likelihood *= (
                probabilities[:, item] if response else 1 - probabilities[:, item]
            )
        masses.append(weights @ likelihood / np.sqrt(np.pi))
    return np.asarray(masses)


def test_configural_affine_likelihood_redundancy_requires_standardized_densities():
    original = TwoParameterLogistic(4)
    original.set_parameters(
        discrimination=np.array([0.7, 1.1, 1.5, 1.9]),
        difficulty=np.array([-1.3, -0.4, 0.3, 1.2]),
    )
    shifted = original.copy()
    scale, location = 1.7, -0.6
    shifted.set_parameters(
        discrimination=original.discrimination / scale,
        difficulty=scale * original.difficulty + location,
    )
    # The item parameters and latent population moments all change materially,
    # while every marginal response-pattern probability stays exactly the same.
    assert_allclose(
        pattern_masses(original, 0.4, 1.2),
        pattern_masses(shifted, scale * 0.4 + location, scale * 1.2),
        rtol=1e-14,
        atol=1e-16,
    )
    model = MultigroupModel(original, 2)
    assert infer_latent_identification(model, InvarianceSpec("configural")) == (
        (False, False),
        (False, False),
    )


def test_metric_shared_slopes_leave_the_latent_mean_unidentified():
    original = TwoParameterLogistic(3)
    original.set_parameters(
        discrimination=np.array([0.8, 1.2, 1.6]), difficulty=np.array([-1.0, 0.2, 1.3])
    )
    shifted = original.copy()
    shifted.set_parameters(difficulty=original.difficulty + 0.9)
    # Translation preserves all shared discriminations and the entire marginal
    # response law, so metric invariance cannot identify a freely varying mean.
    assert_allclose(shifted.discrimination, original.discrimination)
    assert_allclose(
        pattern_masses(original, -0.2, 1.3),
        pattern_masses(shifted, 0.7, 1.3),
        rtol=1e-14,
        atol=1e-16,
    )
    assert infer_latent_identification(
        MultigroupModel(original, 2), InvarianceSpec("metric")
    ) == ((False, False), (False, True))


def test_fixed_scale_and_location_anchor_eliminates_affine_null_directions():
    model = MultigroupModel(TwoParameterLogistic(4), 3)
    model.fix_item_parameters({"discrimination": {0: 1.4}, "difficulty": {0: -0.6}})
    # At the identity transform, a' = a/c and b' = c*b+d. Holding both
    # anchor values fixed gives a full-rank Jacobian in (c,d); fixing only
    # the difficulty leaves a one-dimensional affine equivalence class.
    assert np.linalg.matrix_rank(np.array([[-1.4, 0.0], [-0.6, 1.0]])) == 2
    assert np.linalg.matrix_rank(np.array([[-0.6, 1.0]])) == 1
    assert infer_latent_identification(
        model, InvarianceSpec("configural"), reference_group=1, mean_order=(0, 1, 2)
    ) == ((True, True), (False, False), (True, True))


@pytest.mark.parametrize("level", ["configural", "metric", "scalar", "strict"])
def test_conventional_flags_inspect_prospective_invariance_without_mutation(level):
    model = MultigroupModel(TwoParameterLogistic(4), 2)
    expected = {
        "configural": (False, False),
        "metric": (False, True),
        "scalar": (True, True),
        "strict": (True, True),
    }
    assert infer_latent_identification(model, InvarianceSpec(level)) == (
        (False, False),
        expected[level],
    )
    assert not model.is_item_parameter_shared("discrimination", 0)
    assert not model.is_item_parameter_shared("difficulty", 0)


def test_partial_invariance_uses_remaining_anchor_families():
    model = MultigroupModel(TwoParameterLogistic(4), 2)
    assert infer_latent_identification(
        model, InvarianceSpec("metric", free_discrimination=[0, 1, 2, 3])
    )[1] == (False, False)
    assert infer_latent_identification(
        model, InvarianceSpec("metric", free_discrimination=[0, 1, 2])
    )[1] == (False, True)
    assert infer_latent_identification(
        model, InvarianceSpec("scalar", free_intercepts=[0, 1, 2, 3])
    )[1] == (False, True)
    assert infer_latent_identification(
        model, InvarianceSpec("scalar", free_discrimination=[0, 1, 2, 3])
    )[1] == (True, False)
    assert infer_latent_identification(
        model,
        InvarianceSpec(
            "scalar", free_discrimination=[0, 1, 2, 3], free_intercepts=[0, 1, 2, 3]
        ),
    )[1] == (False, False)
    # A scale anchor and location anchor need not be the same physical item.
    assert infer_latent_identification(
        model,
        InvarianceSpec(
            "scalar", free_discrimination=[0, 1, 2], free_intercepts=[1, 2, 3]
        ),
    )[1] == (True, True)


@pytest.mark.parametrize("model_type", [OneParameterLogistic, PartialCreditModel])
def test_fixed_unit_models_identify_variance_and_need_location_for_free_mean(
    model_type,
):
    base = (
        model_type(3)
        if model_type is OneParameterLogistic
        else model_type(3, [2, 3, 4])
    )
    model = MultigroupModel(base, 2)
    assert infer_latent_identification(
        model, InvarianceSpec("metric", free_discrimination=[0, 1, 2])
    )[1] == (False, True)
    name, value = (
        ("difficulty", 0.3)
        if model_type is OneParameterLogistic
        else ("steps", np.array([-0.5, 0.6, 0.0]))
    )
    model.fix_item_parameters({name: {1: value}})
    assert infer_latent_identification(model, InvarianceSpec("configural"))[1] == (
        True,
        True,
    )


def test_user_fixed_masks_and_registered_rows_are_external_anchors():
    base = GradedResponseModel(3, [2, 3, 4])
    masks = base.free_parameter_masks
    masks["discrimination"][1] = False
    masks["thresholds"][1] = False
    base.set_free_parameter_masks(masks)
    assert infer_latent_identification(
        MultigroupModel(base, 2), InvarianceSpec("configural")
    )[1] == (True, True)
    # A zero slope makes an item independent of theta; fixing its thresholds
    # does not create a latent origin or scale.
    base.set_parameters(discrimination=np.array([1.0, 0.0, 1.0]))
    assert infer_latent_identification(
        MultigroupModel(base, 2), InvarianceSpec("configural")
    )[1] == (False, False)


def test_known_flat_item_does_not_identify_location_despite_shared_difficulty():
    base = TwoParameterLogistic(3)
    base.set_parameters(discrimination=np.array([0.0, 1.0, 1.2]))
    base.set_free_parameter_masks({"discrimination": np.array([False, True, True])})
    shifted = base.copy()
    shifted.set_parameters(difficulty=np.array([0.0, 0.7, 0.7]))
    # Keeping the purported item-0 anchor fixed while translating every
    # informative item's difficulty and the population preserves the full law.
    assert_allclose(
        pattern_masses(base, 0.0, 1.0), pattern_masses(shifted, 0.7, 1.0), rtol=1e-14
    )
    model = MultigroupModel(base, 2)
    assert infer_latent_identification(
        model, InvarianceSpec("scalar", free_intercepts=[1, 2])
    )[1] == (False, True)
    model.fix_item_parameters({"difficulty": {0: 0.0}, "discrimination": {1: 1.0}})
    assert infer_latent_identification(model, InvarianceSpec("configural"))[1] == (
        False,
        False,
    )


@pytest.mark.parametrize(
    "spec",
    [
        InvarianceSpec("configural"),
        InvarianceSpec("metric"),
        InvarianceSpec("scalar", free_intercepts=[0, 1, 2]),
    ],
)
def test_ordered_means_require_identified_location_before_constraints_mutate(spec):
    model = MultigroupModel(TwoParameterLogistic(3), 2)
    with pytest.raises(ValueError, match="identified free means"):
        infer_latent_identification(model, spec, mean_order=(0, 1))
    assert not model.is_item_parameter_shared("difficulty", 0)


def test_focal_fixed_zero_loading_overrides_reference_free_shared_anchor():
    model = MultigroupModel(TwoParameterLogistic(3), 2)
    focal = model.get_group_model(1)
    focal.set_parameters(discrimination=np.array([0.0, 1.0, 1.0]))
    focal.set_free_parameter_masks({"discrimination": np.array([False, True, True])})
    before = [group.parameters for group in model.group_models]
    spec = InvarianceSpec("scalar", free_discrimination=[1, 2], free_intercepts=[1, 2])
    assert infer_latent_identification(model, spec)[1] == (False, False)
    with pytest.raises(ValueError, match="identified free means"):
        infer_latent_identification(model, spec, mean_order=(0, 1))
    for group, original in zip(model.group_models, before, strict=True):
        for name in original:
            assert_allclose(group.parameters[name], original[name])


def test_shared_location_anchor_must_load_in_reference_and_focal_groups():
    model = MultigroupModel(TwoParameterLogistic(3), 3)
    focal = model.get_group_model(2)
    focal.set_parameters(discrimination=np.array([0.0, 1.0, 1.0]))
    focal.set_free_parameter_masks({"discrimination": np.array([False, True, True])})
    # Slopes of item 0 are free by group; its difficulty is the sole location
    # anchor. It identifies group 0 relative to reference group1, but cannot
    # identify the mean of the zero-loading group2.
    spec = InvarianceSpec("scalar", free_discrimination=[0], free_intercepts=[1, 2])
    assert infer_latent_identification(model, spec, reference_group=1) == (
        (True, True),
        (False, False),
        (False, True),
    )


def test_external_single_row_does_not_promote_multidimensional_configural_density():
    base = TwoParameterLogistic(3, n_factors=2)
    model = MultigroupModel(base, 2)
    model.fix_item_parameters(
        {"discrimination": {0: np.array([1.0, 0.0])}, "difficulty": {0: -0.5}}
    )
    assert infer_latent_identification(model, InvarianceSpec("configural"))[1] == (
        False,
        False,
    )
    with pytest.raises(ValueError, match="identified free means"):
        infer_latent_identification(model, InvarianceSpec("scalar"), mean_order=(0, 1))


@pytest.mark.parametrize("reference_group", [True, -0.1, -1, 2])
def test_invalid_reference_groups_do_not_touch_links(reference_group):
    model = MultigroupModel(TwoParameterLogistic(2), 2)
    with pytest.raises(ValueError, match="reference_group"):
        infer_latent_identification(model, InvarianceSpec("scalar"), reference_group)
    assert not model.is_item_parameter_shared("difficulty", 0)


def test_configural_end_to_end_matches_sum_of_separate_standardized_model_likelihoods():
    from mirt import fit_mirt

    rng = np.random.default_rng(90451)
    data = [rng.integers(0, 2, (160, 5)) for _ in range(2)]
    fit = MultigroupEMEstimator(n_quadpts=21, max_iter=150, tol=1e-6).fit(
        MultigroupModel(OneParameterLogistic(5), 2), data, "configural"
    )
    separate = [
        fit_mirt(
            group,
            model="1PL",
            n_quadpts=21,
            max_iter=150,
            tol=1e-6,
            compute_standard_errors=False,
        )
        for group in data
    ]
    assert_allclose(
        fit.log_likelihood, sum(result.log_likelihood for result in separate), atol=3e-3
    )
    assert fit.converged
    assert all(result.converged for result in separate)
    assert fit.n_parameters == 10
    for distribution in fit.latent_distributions:
        assert_allclose(distribution.mean, [0])
        assert_allclose(distribution.cov, [[1]])
