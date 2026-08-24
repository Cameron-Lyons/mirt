"""Correctness contracts for explanatory IRT models."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from mirt.models.explanatory import (
    LLTM,
    ExplanatoryIRT,
    LatentRegressionModel,
    RaschLLTM,
)


@pytest.fixture
def item_features() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.5],
        ]
    )


@pytest.mark.parametrize("model_type", [LLTM, RaschLLTM])
def test_lltm_defensively_owns_structure(
    model_type: Callable[..., LLTM],
    item_features: np.ndarray,
) -> None:
    source = item_features.copy()
    names = ["operation", "content"]
    model = model_type(4, source, feature_names=names)
    source[:] = 0.0
    names[:] = ["changed", "changed"]
    exposed = model.item_features
    exposed[:] = 0.0

    np.testing.assert_array_equal(model.item_features, item_features)
    assert model.feature_names == ["operation", "content"]


@pytest.mark.parametrize(
    ("features", "message"),
    [
        ([1.0, 2.0], "2D"),
        ([[1.0], [2.0]], "rows"),
        (np.empty((4, 0)), "at least one feature"),
        ([[1.0, 0.0], [0.0, 1.0], [1.0, np.nan], [0.0, 0.0]], "finite"),
        ([["yes", "no"]] * 4, "numeric"),
    ],
)
def test_lltm_validates_item_features(features: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LLTM(4, features)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_items", [0, -1, True, 2.5])
def test_lltm_validates_item_count(n_items: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        LLTM(n_items, np.ones((1, 1)))  # type: ignore[arg-type]


def test_lltm_validates_feature_names_and_flag(item_features: np.ndarray) -> None:
    with pytest.raises(ValueError, match="feature_names"):
        LLTM(4, item_features, feature_names=[])
    with pytest.raises(ValueError, match="strings"):
        LLTM(4, item_features, feature_names=["a", 2])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="boolean"):
        LLTM(4, item_features, constrain_discrimination=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("model_type", [LLTM, ExplanatoryIRT])
def test_parameter_setters_validate_and_own_values(
    model_type: Callable[..., LLTM | ExplanatoryIRT],
    item_features: np.ndarray,
) -> None:
    kwargs: dict[str, object] = {}
    if model_type is ExplanatoryIRT:
        kwargs["n_person_covariates"] = 1
    model = model_type(4, item_features, **kwargs)
    weights = np.array([0.4, -0.2])
    model.set_feature_weights(weights)
    weights[:] = 9.0
    exposed = model.feature_weights
    exposed[:] = 8.0

    np.testing.assert_array_equal(model.feature_weights, [0.4, -0.2])
    with pytest.raises(ValueError, match="finite"):
        model.set_feature_weights(np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="positive"):
        model.set_parameters(discrimination=np.array([1.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="common"):
        model.set_parameters(discrimination=np.array([1.0, 1.1, 1.0, 1.0]))


@pytest.mark.parametrize("model_type", [LLTM, ExplanatoryIRT])
def test_unconstrained_models_allow_positive_item_discriminations(
    model_type: Callable[..., LLTM | ExplanatoryIRT],
    item_features: np.ndarray,
) -> None:
    kwargs: dict[str, object] = {"constrain_discrimination": False}
    if model_type is ExplanatoryIRT:
        kwargs["n_person_covariates"] = 1
    model = model_type(4, item_features, **kwargs)
    values = np.array([0.8, 1.0, 1.2, 1.4])

    model.set_parameters(discrimination=values)
    model.set_item_parameter(0, "discrimination", 0.7)

    np.testing.assert_allclose(model.discrimination, [0.7, 1.0, 1.2, 1.4])


@pytest.mark.parametrize("item_idx", [-1, 4, True, 1.5])
def test_lltm_rejects_invalid_item_indices(
    item_features: np.ndarray,
    item_idx: object,
) -> None:
    model = LLTM(4, item_features)

    with pytest.raises(IndexError, match="item_idx"):
        model.probability([[0.0]], item_idx=item_idx)  # type: ignore[arg-type]


def test_lltm_rejects_nonfinite_theta(item_features: np.ndarray) -> None:
    model = LLTM(4, item_features)

    with pytest.raises(ValueError, match="finite"):
        model.probability([[np.nan]])


@pytest.mark.parametrize("n_covariates", [0, -1, True, 1.5])
def test_latent_regression_validates_covariate_count(n_covariates: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        LatentRegressionModel(n_covariates)  # type: ignore[arg-type]


def test_latent_regression_validates_names_and_flag() -> None:
    with pytest.raises(ValueError, match="covariate_names"):
        LatentRegressionModel(2, covariate_names=["one"])
    with pytest.raises(ValueError, match="boolean"):
        LatentRegressionModel(2, include_intercept=1)  # type: ignore[arg-type]


def test_latent_regression_accepts_intuitive_one_dimensional_inputs() -> None:
    one_covariate = LatentRegressionModel(1).set_regression_weights([0.5, 2.0])
    many_covariates = LatentRegressionModel(2).set_regression_weights([0.5, 2.0, -1.0])

    np.testing.assert_allclose(one_covariate.predict_mean([1.0, 2.0]), [2.5, 4.5])
    np.testing.assert_allclose(many_covariates.predict_mean([1.0, 2.0]), [0.5])


@pytest.mark.parametrize(
    ("covariates", "message"),
    [
        ([[[1.0, 2.0]]], "one- or two-dimensional"),
        ([[1.0]], "columns"),
        ([[1.0, np.inf]], "finite"),
        ([["yes", "no"]], "numeric"),
    ],
)
def test_latent_regression_validates_covariate_values(
    covariates: object,
    message: str,
) -> None:
    model = LatentRegressionModel(2)

    with pytest.raises(ValueError, match=message):
        model.predict_mean(covariates)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0, 2.0], "shape"),
        ([0.0, 1.0, np.nan], "finite"),
        (["a", "b", "c"], "numeric"),
    ],
)
def test_latent_regression_validates_weights(
    weights: object,
    message: str,
) -> None:
    model = LatentRegressionModel(2)

    with pytest.raises(ValueError, match=message):
        model.set_regression_weights(weights)  # type: ignore[arg-type]


@pytest.mark.parametrize("variance", [0.0, -1.0, np.nan, np.inf, True])
def test_latent_regression_validates_residual_variance(variance: object) -> None:
    model = LatentRegressionModel(1)

    with pytest.raises(ValueError, match="finite and positive"):
        model.set_residual_variance(variance)  # type: ignore[arg-type]


def test_log_prior_density_normalizes_column_theta_without_outer_broadcast() -> None:
    model = LatentRegressionModel(1).set_regression_weights([0.0, 1.0])
    covariates = np.array([[-1.0], [0.0], [1.0]])

    density = model.log_prior_density(covariates.copy(), covariates)
    scalar_density = model.log_prior_density(0.0, covariates)

    assert density.shape == (3,)
    assert scalar_density.shape == (3,)
    np.testing.assert_allclose(density, -0.5 * np.log(2.0 * np.pi))
    with pytest.raises(ValueError, match="one value or 3"):
        model.log_prior_density(np.array([0.0, 1.0]), covariates)


def test_latent_regression_copy_is_independent() -> None:
    model = LatentRegressionModel(2).set_regression_weights([0.5, 1.0, -0.5])
    model.set_residual_variance(0.4)

    copied = model.copy()
    copied.set_regression_weights([0.0, 0.0, 0.0])
    copied.set_residual_variance(2.0)

    np.testing.assert_allclose(model.regression_weights, [0.5, 1.0, -0.5])
    assert model.residual_variance == 0.4


def test_explanatory_model_validates_names_and_covariate_count(
    item_features: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="feature_names"):
        ExplanatoryIRT(4, item_features, 1, feature_names=["only"])
    with pytest.raises(ValueError, match="covariate_names"):
        ExplanatoryIRT(4, item_features, 2, covariate_names=["only"])
    with pytest.raises(ValueError, match="positive integer"):
        ExplanatoryIRT(4, item_features, 0)


@pytest.mark.parametrize(
    ("residual", "message"),
    [
        ([0.0, 1.0], "one value or 3"),
        ([[[0.0]]], "scalar or one-dimensional"),
        ([0.0, np.nan, 0.0], "finite"),
    ],
)
def test_probability_given_covariates_validates_residual_shape(
    item_features: np.ndarray,
    residual: object,
    message: str,
) -> None:
    model = ExplanatoryIRT(4, item_features, 1)
    covariates = np.array([[-1.0], [0.0], [1.0]])

    with pytest.raises(ValueError, match=message):
        model.probability_given_covariates(
            covariates,
            residual_theta=residual,  # type: ignore[arg-type]
        )


def test_marginal_probability_matches_direct_quadrature(
    item_features: np.ndarray,
) -> None:
    model = ExplanatoryIRT(4, item_features, 1, constrain_discrimination=False)
    model.set_feature_weights(np.array([0.4, -0.2]))
    model.set_regression_weights(np.array([0.3, 0.8]))
    model.set_residual_variance(0.7)
    model.set_parameters(discrimination=np.array([0.7, 1.0, 1.2, 1.5]))
    covariates = np.array([[-1.0], [0.0], [1.0]])
    nodes, weights = hermgauss(15)
    nodes = nodes * np.sqrt(2.0)
    weights = weights / np.sqrt(np.pi)
    mean = model.latent_regression.predict_mean(covariates)
    expected = np.zeros((3, 4))
    for person, person_mean in enumerate(mean):
        theta = person_mean + np.sqrt(0.7) * nodes
        expected[person] = weights @ model.probability(theta[:, None])

    actual = model.marginal_probability_given_covariates(covariates, n_quadpts=15)
    single_item = model.marginal_probability_given_covariates(
        covariates,
        item_idx=2,
        n_quadpts=15,
    )

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(single_item, actual[:, 2])


def test_marginal_probability_is_symmetric_at_zero() -> None:
    model = ExplanatoryIRT(3, np.zeros((3, 1)), 1)
    model.set_regression_weights(np.array([0.0, 0.0]))

    probabilities = model.marginal_probability_given_covariates(
        np.array([[-2.0], [0.0], [3.0]])
    )

    np.testing.assert_allclose(probabilities, 0.5, atol=1e-14)


def test_marginal_log_likelihood_matches_respondent_loop(
    item_features: np.ndarray,
) -> None:
    model = ExplanatoryIRT(4, item_features, 1)
    model.set_feature_weights(np.array([0.2, -0.1]))
    model.set_regression_weights(np.array([0.1, 0.7]))
    model.set_residual_variance(0.6)
    responses = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, -1, 1],
            [1, 1, 0, 0],
        ]
    )
    covariates = np.array([[-1.0], [0.0], [1.0]])
    nodes, weights = hermgauss(17)
    nodes = nodes * np.sqrt(2.0)
    weights = weights / np.sqrt(np.pi)
    mean = model.latent_regression.predict_mean(covariates)
    expected = np.empty(3)
    for person in range(3):
        theta = mean[person] + np.sqrt(0.6) * nodes
        repeated = np.broadcast_to(responses[person], (17, 4))
        conditional = model.log_likelihood(repeated, theta[:, None])
        maximum = np.max(conditional)
        expected[person] = maximum + np.log(weights @ np.exp(conditional - maximum))

    actual = model.marginal_log_likelihood_given_covariates(
        responses,
        covariates,
        n_quadpts=17,
    )

    np.testing.assert_allclose(actual, expected)


def test_marginal_log_likelihood_handles_missing_and_validates_inputs(
    item_features: np.ndarray,
) -> None:
    model = ExplanatoryIRT(4, item_features, 1)
    covariates = np.array([[-1.0], [0.0]])

    likelihood = model.marginal_log_likelihood_given_covariates(
        np.full((2, 4), -9),
        covariates,
    )
    np.testing.assert_allclose(likelihood, 0.0, atol=1e-14)

    with pytest.raises(ValueError, match="two-dimensional"):
        model.marginal_log_likelihood_given_covariates([0, 1], covariates)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only 0 and 1"):
        model.marginal_log_likelihood_given_covariates(
            [[0, 1, 2, 0], [1, 0, 1, 0]],
            covariates,
        )
    with pytest.raises(ValueError, match="same number"):
        model.marginal_log_likelihood_given_covariates(
            [[0, 1, 0, 0]],
            covariates,
        )
    with pytest.raises(ValueError, match="positive integer"):
        model.marginal_log_likelihood_given_covariates(
            np.zeros((2, 4)),
            covariates,
            n_quadpts=0,
        )


def test_rasch_copy_preserves_type_state_and_constraints(
    item_features: np.ndarray,
) -> None:
    model = RaschLLTM(4, item_features).set_feature_weights([0.3, -0.2])
    model._is_fitted = True

    copied = model.copy()
    copied.set_feature_weights([0.0, 0.0])

    assert isinstance(copied, RaschLLTM)
    assert copied.is_fitted
    np.testing.assert_allclose(model.feature_weights, [0.3, -0.2])
    np.testing.assert_allclose(copied.discrimination, 1.0)
    with pytest.raises(ValueError, match="Cannot set one discrimination"):
        copied.set_item_parameter(0, "discrimination", 2.0)
