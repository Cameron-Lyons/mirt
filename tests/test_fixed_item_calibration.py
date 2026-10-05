"""Marginal maximum likelihood fixed-item calibration for any model family."""

import numpy as np
import pytest

from mirt import FixedItemCalibrationResult, fixed_item_calibration, simdata
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel, RatingScaleModel
from mirt.utils.calibration import fixed_calib

ANCHORS = list(range(20))
NEW_ITEMS = list(range(20, 30))


@pytest.fixture(scope="module")
def shifted_2pl():
    """Thirty 2PL items answered by a N(0.5, 1.2^2) calibration sample."""
    rng = np.random.default_rng(0)
    discrimination = rng.uniform(0.8, 2.0, 30)
    difficulty = rng.normal(0.0, 1.0, 30)
    theta = rng.normal(0.5, 1.2, 3000)
    responses = simdata(
        "2PL",
        theta=theta,
        discrimination=discrimination,
        difficulty=difficulty,
        seed=1,
    )
    anchors = TwoParameterLogistic(20).set_parameters(
        discrimination=discrimination[ANCHORS], difficulty=difficulty[ANCHORS]
    )
    return responses, anchors, discrimination, difficulty, theta


def test_anchors_stay_fixed_and_the_population_is_estimated(shifted_2pl):
    responses, anchors, discrimination, difficulty, theta = shifted_2pl

    result = fixed_item_calibration(
        responses, TwoParameterLogistic(30), ANCHORS, anchors
    )

    assert isinstance(result, FixedItemCalibrationResult)
    assert result.anchor_items == ANCHORS
    assert result.new_items == NEW_ITEMS
    assert result.fit_result.converged
    parameters = result.model.parameters
    np.testing.assert_array_equal(
        parameters["discrimination"][ANCHORS], discrimination[ANCHORS]
    )
    np.testing.assert_array_equal(
        parameters["difficulty"][ANCHORS], difficulty[ANCHORS]
    )
    assert result.latent_mean[0] == pytest.approx(theta.mean(), abs=0.06)
    assert np.sqrt(result.latent_cov[0, 0]) == pytest.approx(theta.std(), abs=0.06)
    np.testing.assert_array_equal(
        result.new_item_parameters["difficulty"], parameters["difficulty"][NEW_ITEMS]
    )

    errors = result.fit_result.standard_errors["difficulty"]
    np.testing.assert_array_equal(errors[ANCHORS], 0.0)
    assert np.all(errors[NEW_ITEMS] > 0.0)


def test_mml_calibration_recovers_new_items_better_than_the_heuristic(shifted_2pl):
    responses, anchors, discrimination, difficulty, _ = shifted_2pl

    result = fixed_item_calibration(
        responses,
        TwoParameterLogistic(30),
        ANCHORS,
        anchors,
        compute_standard_errors=False,
    )
    heuristic = fixed_calib(responses, anchors, ANCHORS)

    mml_error = np.abs(
        result.model.parameters["difficulty"][NEW_ITEMS] - difficulty[NEW_ITEMS]
    )
    heuristic_error = np.abs(
        np.asarray(heuristic.new_difficulty) - difficulty[NEW_ITEMS]
    )
    assert np.sqrt(np.mean(mml_error**2)) < np.sqrt(np.mean(heuristic_error**2))
    assert mml_error.max() < 0.15


def test_fixed_population_keeps_the_standard_normal_scale(shifted_2pl):
    responses, anchors, *_ = shifted_2pl

    result = fixed_item_calibration(
        responses,
        TwoParameterLogistic(30),
        ANCHORS,
        anchors,
        estimate_mean=False,
        estimate_cov=False,
        compute_standard_errors=False,
    )

    np.testing.assert_array_equal(result.latent_mean, [0.0])
    np.testing.assert_array_equal(result.latent_cov, [[1.0]])


def test_anchor_values_may_be_given_by_mapping_or_stored_in_the_template(shifted_2pl):
    responses, anchors, discrimination, difficulty, _ = shifted_2pl
    options = {"compute_standard_errors": False}
    template = TwoParameterLogistic(30).set_parameters(
        discrimination=np.where(np.arange(30) < 20, discrimination, 1.0),
        difficulty=np.where(np.arange(30) < 20, difficulty, 0.0),
    )

    from_model = fixed_item_calibration(
        responses, TwoParameterLogistic(30), ANCHORS, anchors, **options
    )
    from_mapping = fixed_item_calibration(
        responses,
        TwoParameterLogistic(30),
        ANCHORS,
        anchors.parameters,
        **options,
    )
    from_template = fixed_item_calibration(responses, template, ANCHORS, **options)

    for result in (from_mapping, from_template):
        for name, values in from_model.model.parameters.items():
            np.testing.assert_array_equal(result.model.parameters[name], values)
    assert template.parameters["difficulty"][25] == 0.0
    # The template itself is copied, not restricted in place.
    assert template.free_parameter_masks["difficulty"].all()


def test_graded_anchors_calibrate_a_shifted_population():
    rng = np.random.default_rng(2)
    discrimination = rng.uniform(0.8, 1.8, 14)
    thresholds = np.sort(rng.normal(0.0, 1.0, (14, 3)), axis=1)
    truth = GradedResponseModel(14, n_categories=4).set_parameters(
        discrimination=discrimination, thresholds=thresholds
    )
    theta = rng.normal(-0.4, 0.8, (2500, 1))
    responses = truth.simulate(theta, seed=4)
    anchors = list(range(9))
    anchor_model = GradedResponseModel(9, n_categories=4).set_parameters(
        discrimination=discrimination[:9], thresholds=thresholds[:9]
    )

    result = fixed_item_calibration(
        responses,
        GradedResponseModel(14, n_categories=4),
        anchors,
        anchor_model,
        compute_standard_errors=False,
    )

    parameters = result.model.parameters
    np.testing.assert_array_equal(parameters["thresholds"][:9], thresholds[:9])
    np.testing.assert_array_equal(parameters["discrimination"][:9], discrimination[:9])
    assert result.latent_mean[0] == pytest.approx(theta.mean(), abs=0.08)
    assert np.sqrt(result.latent_cov[0, 0]) == pytest.approx(theta.std(), abs=0.08)
    assert np.max(np.abs(parameters["thresholds"][9:] - thresholds[9:])) < 0.3


def test_existing_restrictions_on_new_items_are_respected(shifted_2pl):
    responses, anchors, *_ = shifted_2pl
    template = TwoParameterLogistic(30).set_parameters(discrimination=np.full(30, 1.3))
    masks = template.free_parameter_masks
    masks["discrimination"][25] = False
    template.set_free_parameter_masks(masks)

    result = fixed_item_calibration(
        responses, template, ANCHORS, anchors, compute_standard_errors=False
    )

    discrimination = result.model.parameters["discrimination"]
    assert discrimination[25] == 1.3
    assert np.all(discrimination[[20, 21, 22]] != 1.3)


def test_shared_parameters_belong_to_the_anchors():
    responses = simdata("GRM", n_persons=400, n_items=6, n_categories=3, seed=5)
    template = RatingScaleModel(6, n_categories=3)
    shared = template.parameters["thresholds"]

    result = fixed_item_calibration(
        responses,
        template,
        [0, 1],
        {
            "difficulty": np.array([-0.5, 0.5]),
            "thresholds": np.array([0.0, 0.7]),
        },
        compute_standard_errors=False,
    )

    assert not np.array_equal(shared, [0.0, 0.7])
    np.testing.assert_array_equal(result.model.parameters["thresholds"], [0.0, 0.7])
    np.testing.assert_array_equal(
        result.model.parameters["difficulty"][:2], [-0.5, 0.5]
    )
    assert not result.model.free_parameter_masks["thresholds"].any()


def test_family_fixed_anchor_parameters_may_be_omitted():
    responses = simdata("1PL", n_persons=300, n_items=5, seed=6)

    result = fixed_item_calibration(
        responses,
        OneParameterLogistic(5),
        [0, 1, 2],
        {"difficulty": np.array([-1.0, 0.0, 1.0])},
        compute_standard_errors=False,
    )

    np.testing.assert_array_equal(
        result.model.parameters["difficulty"][:3], [-1.0, 0.0, 1.0]
    )


@pytest.mark.parametrize(
    ("anchor_items", "anchor_parameters", "message"),
    [
        ([], None, "anchor_items"),
        ([0, 0], None, "duplicate"),
        ([0, 9], None, "out-of-bounds"),
        (list(range(4)), None, "new item"),
        ([0, 1], TwoParameterLogistic(3), "one item per anchor"),
        ([0, 1], {"slopes": np.ones(2)}, "Unknown anchor parameters"),
        ([0, 1], {"difficulty": np.zeros(2)}, "must provide discrimination"),
        (
            [0, 1],
            {"discrimination": np.ones(3), "difficulty": np.zeros(2)},
            "must have shape",
        ),
        (
            [0, 1],
            {"discrimination": [1.0, np.nan], "difficulty": np.zeros(2)},
            "finite",
        ),
        ([0, 1], "2PL", "mapping"),
    ],
)
def test_rejects_invalid_anchor_specifications(
    anchor_items, anchor_parameters, message
):
    responses = simdata("2PL", n_persons=50, n_items=4, seed=7)

    with pytest.raises(MirtValidationError, match=message):
        fixed_item_calibration(
            responses, TwoParameterLogistic(4), anchor_items, anchor_parameters
        )


def test_rejects_templates_without_free_new_item_parameters():
    responses = simdata("2PL", n_persons=50, n_items=3, seed=8)
    template = TwoParameterLogistic(3)
    template.set_free_parameter_masks(
        {name: np.zeros(3, dtype=bool) for name in template.parameters}
    )

    with pytest.raises(MirtValidationError, match="no free parameters"):
        fixed_item_calibration(responses, template, [0])
    with pytest.raises(MirtValidationError, match="estimate_mean"):
        fixed_item_calibration(responses, TwoParameterLogistic(3), [0], estimate_mean=1)
    with pytest.raises(MirtValidationError, match="item model"):
        fixed_item_calibration(responses, "2PL", [0])


def test_rejects_anchor_models_with_different_category_counts():
    responses = simdata("GRM", n_persons=60, n_items=4, n_categories=4, seed=9)
    anchors = GradedResponseModel(2, n_categories=[3, 4])

    with pytest.raises(MirtValidationError, match="same category counts"):
        fixed_item_calibration(
            responses, GradedResponseModel(4, n_categories=4), [0, 1], anchors
        )
    with pytest.raises(MirtValidationError, match="same category counts"):
        fixed_item_calibration(
            responses,
            GradedResponseModel(4, n_categories=4),
            [0, 1],
            TwoParameterLogistic(2),
        )
