import numpy as np
import pytest

from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models import TwoParameterLogistic
from mirt.utils import calibration as calibration_module
from mirt.utils import fixed_calib


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
    theta_grid = np.linspace(-4.0, 4.0, n_quadpts)
    log_weights = -0.5 * theta_grid**2
    log_weights -= np.log(np.exp(log_weights).sum())

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
        use_rust=False,
        max_iter=8,
    )
    nan_result = fixed_calib(
        nan_responses,
        anchor_model,
        anchor_items=[0, 1],
        new_items=[2, 3],
        use_rust=False,
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


@pytest.mark.parametrize("use_rust", [False, True])
def test_returned_scores_and_likelihood_use_final_parameters(
    calibration_data, use_rust
):
    if use_rust and not calibration_module.RUST_AVAILABLE:
        pytest.skip("native backend is not installed")

    responses, anchor_model = calibration_data
    result = fixed_calib(
        responses,
        anchor_model,
        anchor_items=[0, 1],
        new_items=[2, 3],
        n_quadpts=15,
        max_iter=1,
        tol=1e-12,
        use_rust=use_rust,
    )
    expected_theta, expected_log_likelihood = _expected_final_state(
        responses, anchor_model, result, n_quadpts=15
    )

    np.testing.assert_allclose(result.theta, expected_theta, rtol=1e-10, atol=1e-10)
    assert result.log_likelihood == pytest.approx(
        expected_log_likelihood, rel=1e-10, abs=1e-10
    )
    assert result.n_iterations == 1


def test_native_and_python_results_match(calibration_data):
    if not calibration_module.RUST_AVAILABLE:
        pytest.skip("native backend is not installed")

    responses, anchor_model = calibration_data
    python_result = fixed_calib(
        responses,
        anchor_model,
        anchor_items=[0, 1],
        use_rust=False,
        max_iter=10,
    )
    native_result = fixed_calib(
        responses,
        anchor_model,
        anchor_items=[0, 1],
        use_rust=True,
        max_iter=10,
    )

    np.testing.assert_allclose(
        native_result.new_discrimination,
        python_result.new_discrimination,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        native_result.new_difficulty,
        python_result.new_difficulty,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        native_result.theta, python_result.theta, rtol=1e-10, atol=1e-10
    )
    assert native_result.log_likelihood == pytest.approx(
        python_result.log_likelihood, rel=1e-10, abs=1e-10
    )


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
        min_count=0,
        min_valid_points=2,
        use_rust=False,
    )

    assert np.isfinite(result.log_likelihood)
    assert np.all(np.isfinite(result.theta))


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
        fixed_calib(responses, anchor_model, [0], [1], use_rust=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_type": "3PL"}, "only the 2PL"),
        ({"max_iter": 0}, "max_iter"),
        ({"n_quadpts": 1}, "n_quadpts"),
        ({"tol": 0}, "tol"),
        ({"tol": "fast"}, "finite number"),
        ({"disc_bounds": (0, 2)}, "strictly positive"),
        ({"prob_clamp": (0, 0.9)}, "strictly between"),
        ({"min_valid_points": 22}, "must not exceed"),
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
            use_rust=False,
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
            use_rust=False,
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
            use_rust=False,
        )
