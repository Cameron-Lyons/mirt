"""Tests for uncertainty-integrated mixed-effects predictions."""

from types import SimpleNamespace

import numpy as np
import pytest

from mirt import MixedEffectsFitResult, predict_mixed
from mirt.models import GradedResponseModel, TwoParameterLogistic
from mirt.utils import predictions as prediction_utils


def _binary_result() -> MixedEffectsFitResult:
    model = TwoParameterLogistic(n_items=3)
    model.set_parameters(
        discrimination=np.array([0.8, 1.2, 1.6]),
        difficulty=np.array([-0.7, 0.1, 0.9]),
    )
    return MixedEffectsFitResult(
        model=model,
        person_effects=np.array([0.7, -0.25]),
        item_effects=None,
        log_likelihood=-10.0,
        aic=30.0,
        bic=35.0,
        converged=True,
        residual_variance=0.36,
        theta=np.array([-1.2, -0.1, 0.8, 1.5]),
        theta_se=np.array([0.1, 0.25, 0.4, 0.7]),
        person_intercept=0.2,
        person_covariate_names=("x", "z"),
    )


def _scalar_reference(
    model,
    theta: np.ndarray,
    standard_errors: np.ndarray,
    n_quadpts: int,
    item_idx: int | None = None,
) -> np.ndarray:
    nodes, weights = np.polynomial.hermite.hermgauss(n_quadpts)
    nodes *= np.sqrt(2.0)
    weights /= np.sqrt(np.pi)
    values = []
    for center, standard_error in zip(theta, standard_errors, strict=True):
        probabilities = model.probability(
            (center + standard_error * nodes)[:, None],
            item_idx=item_idx,
        )
        values.append(np.tensordot(weights, probabilities, axes=([0], [0])))
    return np.asarray(values, dtype=np.float64)


def test_stored_uncertainty_matches_scalar_integration_and_auto_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _binary_result()
    expected = _scalar_reference(result.model, result.theta, result.theta_se, 17)

    explicit_chunks = predict_mixed(
        result,
        integrate_uncertainty=True,
        n_quadpts=17,
        chunk_size=2,
    )
    monkeypatch.setattr(
        prediction_utils,
        "_MIXED_PREDICTION_MAX_PROBABILITY_VALUES",
        52,
    )
    automatic_chunks = predict_mixed(
        result,
        integrate_uncertainty=True,
        n_quadpts=17,
    )

    np.testing.assert_allclose(explicit_chunks, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(automatic_chunks, expected, rtol=1e-14, atol=1e-14)


def test_zero_uncertainty_reduces_to_plugin_predictions() -> None:
    result = _binary_result()
    plugin = predict_mixed(result)
    integrated = predict_mixed(
        result,
        integrate_uncertainty=True,
        standard_errors=0.0,
        chunk_size=1,
    )

    np.testing.assert_allclose(integrated, plugin, rtol=1e-14, atol=1e-14)


def test_explicit_abilities_accept_scalar_uncertainty() -> None:
    result = _binary_result()
    theta = np.array([-1.0, 0.0, 1.0])
    expected = _scalar_reference(result.model, theta, np.full(3, 0.3), 13)

    actual = predict_mixed(
        result,
        new_theta=theta,
        integrate_uncertainty=True,
        standard_errors=0.3,
        n_quadpts=13,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_covariate_predictions_default_to_residual_uncertainty() -> None:
    result = _binary_result()
    covariates = np.array([[1.0, -1.0], [0.0, 2.0], [2.0, 0.5]])

    default = predict_mixed(
        result,
        new_covariates=covariates,
        integrate_uncertainty=True,
    )
    explicit = predict_mixed(
        result,
        new_covariates=covariates,
        integrate_uncertainty=True,
        standard_errors=np.sqrt(result.residual_variance),
    )

    np.testing.assert_allclose(default, explicit, rtol=0.0, atol=0.0)


def test_polytomous_integration_preserves_shapes_and_probability_mass() -> None:
    model = GradedResponseModel(n_items=2, n_categories=[3, 4])
    theta = np.array([-0.8, 0.4, 1.1])
    standard_errors = np.array([0.1, 0.3, 0.6])
    result = SimpleNamespace(model=model, theta=theta, theta_se=standard_errors)

    full = predict_mixed(
        result,
        integrate_uncertainty=True,
        n_quadpts=15,
        chunk_size=2,
    )
    selected = predict_mixed(
        result,
        item_idx=0,
        integrate_uncertainty=True,
        n_quadpts=15,
        chunk_size=2,
    )

    assert full.shape == (3, 2, 4)
    assert selected.shape == (3, 3)
    np.testing.assert_allclose(full[:, 0, :3].sum(axis=1), 1.0)
    np.testing.assert_allclose(full[:, 1, :4].sum(axis=1), 1.0)
    np.testing.assert_allclose(full[:, 0, 3], 0.0)
    np.testing.assert_allclose(
        full,
        _scalar_reference(model, theta, standard_errors, 15),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(selected, full[:, 0, :3], rtol=1e-14, atol=1e-14)


def test_selected_item_path_never_builds_full_probability_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _binary_result()
    calls: list[int | None] = []
    probability = result.model.probability

    def tracked_probability(theta, item_idx=None):
        calls.append(item_idx)
        return probability(theta, item_idx=item_idx)

    monkeypatch.setattr(result.model, "probability", tracked_probability)

    selected = predict_mixed(
        result,
        item_idx=1,
        integrate_uncertainty=True,
        chunk_size=2,
    )

    assert selected.shape == (4,)
    assert calls == [1, 1]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"standard_errors": 0.2}, ValueError, "requires"),
        (
            {"integrate_uncertainty": True, "new_theta": np.array([0.0])},
            ValueError,
            "required",
        ),
        (
            {"integrate_uncertainty": True, "standard_errors": [-0.1] * 4},
            ValueError,
            "non-negative",
        ),
        (
            {"integrate_uncertainty": True, "standard_errors": [np.nan] * 4},
            ValueError,
            "finite",
        ),
        (
            {"integrate_uncertainty": True, "standard_errors": [0.1, 0.2]},
            ValueError,
            "4 values",
        ),
        ({"integrate_uncertainty": True, "n_quadpts": 1}, ValueError, "at least 2"),
        ({"integrate_uncertainty": True, "n_quadpts": 3.5}, ValueError, "integer"),
        ({"integrate_uncertainty": True, "n_quadpts": True}, ValueError, "integer"),
        ({"integrate_uncertainty": True, "chunk_size": 0}, ValueError, "positive"),
        ({"integrate_uncertainty": True, "chunk_size": 2.5}, ValueError, "integer"),
        ({"integrate_uncertainty": True, "chunk_size": True}, ValueError, "integer"),
        ({"integrate_uncertainty": "yes"}, TypeError, "boolean"),
    ],
)
def test_uncertainty_integration_validates_controls(
    kwargs: dict,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        predict_mixed(_binary_result(), **kwargs)


def test_uncertainty_integration_requires_available_valid_uncertainty() -> None:
    result = _binary_result()
    result.theta_se = None
    with pytest.raises(ValueError, match="does not contain"):
        predict_mixed(result, integrate_uncertainty=True)

    result.residual_variance = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        predict_mixed(
            result,
            new_covariates=np.array([[0.0, 1.0]]),
            integrate_uncertainty=True,
        )


def test_uncertainty_integration_rejects_multidimensional_models() -> None:
    model = SimpleNamespace(
        n_factors=2,
        n_items=1,
        probability=lambda theta, item_idx=None: np.zeros((len(theta), 1)),
    )
    result = SimpleNamespace(model=model, theta=np.array([[0.0, 0.0]]), theta_se=[0.1])

    with pytest.raises(ValueError, match="unidimensional"):
        predict_mixed(result, integrate_uncertainty=True)
