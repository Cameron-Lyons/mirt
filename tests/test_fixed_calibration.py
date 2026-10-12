import numpy as np
import pytest
from scipy.special import expit

from mirt.estimation.quadrature import build_quadrature
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models import TwoParameterLogistic
from mirt.utils.calibration import fixed_calib, fixed_item_calibration


@pytest.fixture
def calibration_data():
    rng = np.random.default_rng(413)
    theta = rng.normal(size=300)
    discrimination = np.array([1.3, 0.8, 1.1, 1.5])
    difficulty = np.array([-0.6, 0.4, -0.2, 0.7])
    probability = 1.0 / (
        1.0 + np.exp(-discrimination[None, :] * (theta[:, None] - difficulty[None, :]))
    )
    responses = (rng.random(probability.shape) < probability).astype(float)
    anchor_model = TwoParameterLogistic(n_items=2).set_parameters(
        discrimination=discrimination[:2],
        difficulty=difficulty[:2],
    )
    return responses, anchor_model


def _expected_final_state(responses, anchor_model, result, n_quadpts):
    normalized = np.where(np.isnan(responses) | (responses < 0), -1, responses)
    quadrature = build_quadrature(n_quadpts, 1)
    theta_grid = quadrature.nodes[:, 0]
    log_weights = np.log(quadrature.weights)

    discrimination = np.concatenate(
        [anchor_model.discrimination, result.new_discrimination]
    )
    difficulty = np.concatenate([anchor_model.difficulty, result.new_difficulty])
    logits = discrimination[None, :] * (theta_grid[:, None] - difficulty[None, :])
    log_probability = -np.logaddexp(0.0, -logits)
    log_complement = -np.logaddexp(0.0, logits)
    correct = (normalized == 1).astype(float)
    incorrect = (normalized == 0).astype(float)
    log_joint = correct @ log_probability.T + incorrect @ log_complement.T + log_weights
    row_maximum = log_joint.max(axis=1, keepdims=True)
    scaled = np.exp(log_joint - row_maximum)
    posterior = scaled / scaled.sum(axis=1, keepdims=True)
    log_likelihood = np.sum(row_maximum[:, 0] + np.log(scaled.sum(axis=1)))
    return posterior @ theta_grid, log_likelihood


@pytest.fixture(scope="module")
def extreme_item_data():
    """Twenty 2PL items, including new items with difficulties near -2.8."""
    rng = np.random.default_rng(3)
    discrimination = rng.lognormal(0.2, 0.3, 20)
    difficulty = rng.normal(0.0, 1.0, 20)
    theta = rng.standard_normal(3000)
    probability = expit(discrimination * (theta[:, None] - difficulty))
    responses = (rng.random(probability.shape) < probability).astype(float)
    anchor_model = TwoParameterLogistic(n_items=10).set_parameters(
        discrimination=discrimination[:10], difficulty=difficulty[:10]
    )
    return responses, anchor_model


def test_new_items_match_r_mirt_fixed_population_mml(extreme_item_data):
    responses, anchor_model = extreme_item_data

    result = fixed_calib(responses, anchor_model, list(range(10)))

    # R mirt 1.48: mirt(y, 1, "2PL", pars = sv, quadpts = 21, TOL = 1e-8) with
    # the anchors and the N(0, 1) group parameters fixed, which is the
    # estimator of fixedCalib(PAU = "NWU"). The former regression M-step put
    # item 17 (true a = 1.41, b = -2.83) at a = 0.87, b = -4.40.
    r_discrimination = [
        1.18336, 1.06312, 1.15372, 1.01077, 0.86023,
        1.02555, 1.38065, 1.17776, 1.59047, 1.02641,
    ]  # fmt: skip
    r_difficulty = [
        -1.00472, -0.30814, 0.83909, 0.59071, 0.10612,
        0.70319, -2.83582, 0.95155, -0.95027, -1.86618,
    ]  # fmt: skip
    np.testing.assert_allclose(result.new_discrimination, r_discrimination, atol=1e-3)
    np.testing.assert_allclose(result.new_difficulty, r_difficulty, atol=1e-3)
    assert result.converged
    # R reports logLik -32592.3514 on the same 21-point grid.
    assert result.log_likelihood == pytest.approx(-32592.3514, abs=1e-3)


def test_fixed_calib_is_fixed_population_item_calibration(calibration_data):
    responses, anchor_model = calibration_data

    result = fixed_calib(responses, anchor_model, [0, 1], tol=1e-6)
    reference = fixed_item_calibration(
        responses,
        TwoParameterLogistic(n_items=4),
        [0, 1],
        anchor_model,
        estimate_mean=False,
        estimate_cov=False,
        tol=1e-6,
        compute_standard_errors=False,
    )

    parameters = reference.new_item_parameters
    np.testing.assert_allclose(
        result.new_discrimination, parameters["discrimination"], rtol=1e-12
    )
    np.testing.assert_allclose(
        result.new_difficulty, parameters["difficulty"], rtol=1e-12
    )
    assert result.log_likelihood == pytest.approx(
        reference.fit_result.log_likelihood, rel=1e-12
    )
    assert result.n_iterations == reference.fit_result.n_iterations


def test_new_item_subset_ignores_other_columns(calibration_data):
    responses, anchor_model = calibration_data

    subset = fixed_calib(responses, anchor_model, [0, 1], new_items=[3])
    reduced = fixed_calib(responses[:, [0, 1, 3]], anchor_model, [0, 1])

    np.testing.assert_array_equal(subset.new_discrimination, reduced.new_discrimination)
    np.testing.assert_array_equal(subset.new_difficulty, reduced.new_difficulty)
    np.testing.assert_array_equal(subset.theta, reduced.theta)


@pytest.mark.parametrize(
    "kwargs",
    [{"prob_clamp": (0.01, 0.99)}, {"min_count": 1.0}, {"min_valid_points": 3}],
)
def test_regression_controls_are_deprecated_and_ignored(calibration_data, kwargs):
    responses, anchor_model = calibration_data
    expected = fixed_calib(responses, anchor_model, [0, 1])

    with pytest.warns(DeprecationWarning, match="deprecated and ignored"):
        result = fixed_calib(responses, anchor_model, [0, 1], **kwargs)

    np.testing.assert_array_equal(result.new_difficulty, expected.new_difficulty)


def test_negative_and_nan_missing_codes_are_equivalent(calibration_data):
    responses, anchor_model = calibration_data
    responses[::7, 0] = -9
    responses[::11, 2] = -1
    nan_responses = responses.copy()
    nan_responses[nan_responses < 0] = np.nan

    negative_result = fixed_calib(
        responses,
        anchor_model,
        anchor_items=[0, 1],
        new_items=[2, 3],
        max_iter=8,
    )
    nan_result = fixed_calib(
        nan_responses,
        anchor_model,
        anchor_items=[0, 1],
        new_items=[2, 3],
        max_iter=8,
    )

    np.testing.assert_allclose(
        negative_result.new_discrimination, nan_result.new_discrimination
    )
    np.testing.assert_allclose(
        negative_result.new_difficulty, nan_result.new_difficulty
    )
    np.testing.assert_allclose(negative_result.theta, nan_result.theta)
    assert negative_result.log_likelihood == pytest.approx(nan_result.log_likelihood)


@pytest.mark.parametrize("n_quadpts", [7, 15])
def test_returned_scores_and_likelihood_use_final_parameters(
    calibration_data, n_quadpts
):
    responses, anchor_model = calibration_data
    result = fixed_calib(
        responses,
        anchor_model,
        anchor_items=[0, 1],
        new_items=[2, 3],
        n_quadpts=n_quadpts,
        max_iter=1,
        tol=1e-12,
    )
    expected_theta, expected_log_likelihood = _expected_final_state(
        responses, anchor_model, result, n_quadpts=n_quadpts
    )

    np.testing.assert_allclose(result.theta, expected_theta, rtol=1e-10, atol=1e-10)
    assert result.log_likelihood == pytest.approx(
        expected_log_likelihood, rel=1e-10, abs=1e-10
    )
    assert result.n_iterations == 1


def test_use_rust_does_not_change_the_estimator(calibration_data):
    responses, anchor_model = calibration_data
    python_result = fixed_calib(
        responses, anchor_model, anchor_items=[0, 1], use_rust=False
    )
    native_result = fixed_calib(responses, anchor_model, anchor_items=[0, 1])

    np.testing.assert_array_equal(
        native_result.new_discrimination, python_result.new_discrimination
    )
    np.testing.assert_array_equal(
        native_result.new_difficulty, python_result.new_difficulty
    )
    np.testing.assert_array_equal(native_result.theta, python_result.theta)


def test_extreme_anchor_likelihood_remains_finite():
    responses = np.ones((4, 11))
    anchor_model = TwoParameterLogistic(n_items=10).set_parameters(
        discrimination=np.full(10, 5.0),
        difficulty=np.full(10, 20.0),
    )

    result = fixed_calib(
        responses,
        anchor_model,
        anchor_items=list(range(10)),
        new_items=[10],
        n_quadpts=5,
        max_iter=1,
    )

    assert np.isfinite(result.log_likelihood)
    assert np.all(np.isfinite(result.theta))
    # Every person answers correctly, so the estimate is clipped to the bound.
    np.testing.assert_array_equal(result.new_difficulty, [-5.0])


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (np.ones(3), "2D"),
        (np.empty((0, 3)), "at least one person"),
        (np.array([[0.0, 2.0, 1.0]]), "binary responses"),
        (np.array([[0.0, np.inf, 1.0]]), "infinite"),
    ],
)
def test_invalid_responses_raise_data_error(responses, message):
    anchor_model = TwoParameterLogistic(n_items=1)

    with pytest.raises(MirtDataError, match=message):
        fixed_calib(responses, anchor_model, [0], [1])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_type": "3PL"}, "only the 2PL"),
        ({"max_iter": 0}, "max_iter"),
        ({"n_quadpts": 4}, "n_quadpts"),
        ({"tol": 0}, "tol"),
        ({"tol": "fast"}, "finite number"),
        ({"disc_bounds": (0, 2)}, "strictly positive"),
        ({"init_disc": 6.0}, "within disc_bounds"),
        ({"use_rust": "yes"}, "use_rust"),
    ],
)
def test_invalid_controls_raise_validation_error(calibration_data, kwargs, message):
    responses, anchor_model = calibration_data

    with pytest.raises(MirtValidationError, match=message):
        fixed_calib(
            responses,
            anchor_model,
            [0, 1],
            [2, 3],
            **kwargs,
        )


@pytest.mark.parametrize(
    ("anchor_items", "new_items", "message"),
    [
        ([0, 0], [2, 3], "duplicate"),
        ([0, 4], [2, 3], "out-of-bounds"),
        ([0, 1], [1, 3], "disjoint"),
        ([0, 1], [], "at least one item"),
    ],
)
def test_invalid_item_sets_raise_validation_error(
    calibration_data, anchor_items, new_items, message
):
    responses, anchor_model = calibration_data

    with pytest.raises(MirtValidationError, match=message):
        fixed_calib(
            responses,
            anchor_model,
            anchor_items,
            new_items,
        )


def test_anchor_model_size_must_match_anchor_items(calibration_data):
    responses, _ = calibration_data
    anchor_model = TwoParameterLogistic(n_items=3)

    with pytest.raises(MirtValidationError, match="one parameter set"):
        fixed_calib(
            responses,
            anchor_model,
            anchor_items=[0, 1],
            new_items=[2, 3],
        )
