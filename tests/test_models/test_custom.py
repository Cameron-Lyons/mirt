"""Tests for callback-based item and group models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import mirt
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.base import BaseItemModel
from mirt.models.custom import (
    LOGISTIC_DEVIATION,
    STANDARD_2PL,
    CustomGroupModel,
    CustomItemModel,
    GroupSpec,
    ItemTypeSpec,
    create_group,
    create_item_type,
    createGroup,
    get_standard_item_type,
    list_standard_item_types,
)


def logistic(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


@pytest.fixture
def logistic_spec() -> ItemTypeSpec:
    return create_item_type(
        "Logistic",
        logistic,
        par_bounds={"a": (0.1, 4.0), "b": (-3.0, 3.0)},
        par_defaults={"a": 1.0, "b": 0.0},
    )


def ordinal(theta: np.ndarray, shift: float) -> np.ndarray:
    eta = theta - shift
    weights = np.column_stack((np.ones_like(eta), np.exp(eta), np.exp(2.0 * eta)))
    return weights / weights.sum(axis=1, keepdims=True)


@pytest.fixture
def ordinal_spec() -> ItemTypeSpec:
    return create_item_type(
        "Ordinal",
        ordinal,
        par_bounds={"shift": (-3.0, 3.0)},
        par_defaults={"shift": 0.0},
        n_categories=3,
    )


def test_item_spec_infers_parameters_and_copies_metadata() -> None:
    names = ["a", "b"]
    bounds = {"a": (0.1, 4.0)}
    defaults = {"a": 1.0}
    spec = create_item_type(
        " Logistic ",
        logistic,
        par_names=names,
        par_bounds=bounds,
        par_defaults=defaults,
    )

    names.append("extra")
    bounds["a"] = (2.0, 3.0)
    defaults["a"] = 2.0

    assert spec.name == "Logistic"
    assert spec.par_names == ["a", "b"]
    assert spec.par_bounds == {"a": (0.1, 4.0), "b": (-np.inf, np.inf)}
    assert spec.par_defaults == {"a": 1.0, "b": 0.0}
    assert create_item_type("inferred", logistic).par_names == ["a", "b"]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ItemTypeSpec("", logistic),
        lambda: ItemTypeSpec("x", None),  # type: ignore[arg-type]
        lambda: ItemTypeSpec("x", logistic, info_function=1),  # type: ignore[arg-type]
        lambda: ItemTypeSpec("x", logistic, gradient_function=1),  # type: ignore[arg-type]
        lambda: ItemTypeSpec("x", logistic, n_categories=1),
        lambda: ItemTypeSpec("x", logistic, par_names=["a", "a"]),
        lambda: ItemTypeSpec("x", logistic, par_names=[""]),
        lambda: ItemTypeSpec("x", logistic, par_bounds={"a": (0.0, 1.0)}),
        lambda: ItemTypeSpec("x", logistic, par_defaults={"a": 0.0}),
        lambda: ItemTypeSpec(
            "x", logistic, par_names=["a"], par_bounds={"a": (1.0, 0.0)}
        ),
        lambda: ItemTypeSpec(
            "x", logistic, par_names=["a"], par_bounds={"a": (0.0, 1.0, 2.0)}
        ),
        lambda: ItemTypeSpec(
            "x",
            logistic,
            par_names=["a"],
            par_bounds={"a": (0.1, 1.0)},
        ),
        lambda: create_item_type("x", lambda: np.array([0.5])),
    ],
)
def test_invalid_item_specs_fail_early(factory: Callable[[], object]) -> None:
    with pytest.raises(MirtValidationError):
        factory()


def test_dichotomous_probability_information_and_scores(
    logistic_spec: ItemTypeSpec,
) -> None:
    model = CustomItemModel(2, logistic_spec, item_names=["a", "b"])
    model.set_parameters(a=[1.0, 2.0], b=[0.0, 0.5])
    theta = np.array([-1.0, 0.0, 1.0])

    assert isinstance(model, BaseItemModel)
    assert model.item_names == ["a", "b"]
    assert model.model_name == "Logistic"
    assert model.n_parameters == 4
    assert not model.is_polytomous
    assert model.n_categories == 2
    np.testing.assert_allclose(model.probability(theta, 0), logistic(theta, 1.0, 0.0))
    assert model.probability(theta).shape == (3, 2)
    np.testing.assert_allclose(model.icc(theta, 0), model.probability(theta, 0))
    np.testing.assert_allclose(
        model.category_probability(theta, 0, 0), 1.0 - model.probability(theta, 0)
    )
    np.testing.assert_allclose(
        model.expected_score(theta), model.probability(theta).sum(axis=1)
    )

    probability = model.probability(theta, 0)
    expected_information = probability * (1.0 - probability)
    np.testing.assert_allclose(
        model.information(theta, 0), expected_information, rtol=1e-8, atol=1e-8
    )
    assert model.information(theta).shape == (3, 2)
    assert model.probability(0.0, 0).shape == (1,)


def test_analytical_information_is_used() -> None:
    spec = create_item_type(
        "with info",
        logistic,
        info_function=lambda theta, a, b: np.full_like(theta, a + b),
        par_names=["a", "b"],
        par_defaults={"a": 1.0, "b": 1.0},
    )
    model = CustomItemModel(1, spec)
    np.testing.assert_array_equal(
        model.information(np.array([0.0, 1.0]), 0), [2.0, 2.0]
    )


def test_multidimensional_callback_receives_full_theta() -> None:
    spec = create_item_type(
        "multidimensional",
        lambda theta, a: 1.0 / (1.0 + np.exp(-a * theta.sum(axis=1))),
        par_names=["a"],
        par_bounds={"a": (0.1, 3.0)},
        par_defaults={"a": 1.0},
    )
    model = CustomItemModel(1, spec, n_factors=2)
    theta = np.array([[0.0, 0.0], [1.0, 2.0]])

    probabilities = model.probability(theta, 0)
    np.testing.assert_allclose(probabilities, [0.5, 1.0 / (1.0 + np.exp(-3.0))])
    expected = 2.0 * probabilities * (1.0 - probabilities)
    np.testing.assert_allclose(model.information(theta, 0), expected, rtol=1e-7)


def test_polytomous_probabilities_scores_information_and_likelihood(
    ordinal_spec: ItemTypeSpec,
) -> None:
    model = CustomItemModel(2, ordinal_spec).set_parameters(shift=[0.0, 0.5])
    theta = np.array([-1.0, 0.0, 1.0])
    probabilities = model.probability(theta)

    assert model.is_polytomous
    assert probabilities.shape == (3, 2, 3)
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0)
    np.testing.assert_allclose(
        model.category_probability(theta, 1, 2), probabilities[:, 1, 2]
    )
    scores = np.arange(3)
    np.testing.assert_allclose(
        model.expected_score(theta, 0), probabilities[:, 0] @ scores
    )
    np.testing.assert_allclose(
        model.expected_score(theta), (probabilities @ scores).sum(axis=1)
    )
    assert model.information(theta).shape == (3, 2)
    assert np.all(model.information(theta) >= 0.0)

    responses = np.array([[0, 2], [1, -1], [2, 0]])
    expected = np.array(
        [
            np.log(probabilities[0, 0, 0]) + np.log(probabilities[0, 1, 2]),
            np.log(probabilities[1, 0, 1]),
            np.log(probabilities[2, 0, 2]) + np.log(probabilities[2, 1, 0]),
        ]
    )
    np.testing.assert_allclose(model.log_likelihood(responses, theta), expected)


@pytest.mark.parametrize("polytomous", [False, True])
def test_batch_likelihood_matches_individual_evaluation(
    logistic_spec: ItemTypeSpec,
    ordinal_spec: ItemTypeSpec,
    polytomous: bool,
) -> None:
    model = CustomItemModel(2, ordinal_spec if polytomous else logistic_spec)
    responses = (
        np.array([[0, 2], [1, -1], [2, 1]])
        if polytomous
        else np.array([[0, 1], [1, -1], [1, 0]])
    )
    theta = np.array([-1.0, 0.0, 1.0])

    batch = model.log_likelihood_batch(responses, theta)
    expected = np.column_stack(
        [
            model.log_likelihood(responses, np.full(responses.shape[0], point))
            for point in theta
        ]
    )
    np.testing.assert_allclose(batch, expected)


def test_parameter_updates_are_bounded_atomic_and_independent(
    logistic_spec: ItemTypeSpec,
) -> None:
    model = CustomItemModel(2, logistic_spec)
    assert model.set_parameters(a=2.0) is model
    np.testing.assert_array_equal(model.parameters["a"], [2.0, 2.0])

    snapshot = model.parameters
    snapshot["a"][0] = 99.0
    assert model.parameters["a"][0] == 2.0

    with pytest.raises(MirtValidationError):
        model.set_parameters(a=[1.0, 2.0], b=[0.0, 99.0])
    np.testing.assert_array_equal(model.parameters["a"], [2.0, 2.0])
    np.testing.assert_array_equal(model.parameters["b"], [0.0, 0.0])

    model.set_item_parameter(1, "b", 0.75)
    assert model.get_item_parameters(1)["b"] == 0.75

    for action in (
        lambda: model.set_parameters(nope=1.0),
        lambda: model.set_parameters(a=[1.0]),
        lambda: model.set_item_parameter(3, "a", 1.0),
        lambda: model.set_item_parameter(0, "nope", 1.0),
        lambda: model.set_item_parameter(0, "a", [1.0]),
        lambda: model.set_item_parameter(0, "a", 99.0),
    ):
        with pytest.raises((MirtValidationError, IndexError)):
            action()


def test_item_model_copy_is_independent(logistic_spec: ItemTypeSpec) -> None:
    model = CustomItemModel(1, logistic_spec, item_names=["one"]).set_parameters(a=2.0)
    model._is_fitted = True
    copied = model.copy()
    copied.set_item_parameter(0, "a", 3.0)

    assert copied.item_names == ["one"]
    assert copied.is_fitted
    assert model.parameters["a"][0] == 2.0
    assert copied.parameters["a"][0] == 3.0


def test_parameter_gradients_use_callback_or_numerical_fallback() -> None:
    analytic = create_item_type(
        "analytic",
        logistic,
        gradient_function=lambda theta, a, b: {
            "a": (theta - b) * logistic(theta, a, b) * (1 - logistic(theta, a, b)),
            "b": -a * logistic(theta, a, b) * (1 - logistic(theta, a, b)),
        },
        par_names=["a", "b"],
        par_defaults={"a": 1.0, "b": 0.0},
    )
    theta = np.array([-1.0, 0.0, 1.0])
    analytic_gradient = CustomItemModel(1, analytic).parameter_gradient(theta, 0)
    numerical_gradient = CustomItemModel(
        1,
        create_item_type(
            "numeric",
            logistic,
            par_names=["a", "b"],
            par_defaults={"a": 1.0, "b": 0.0},
        ),
    ).parameter_gradient(theta, 0)

    for name in ("a", "b"):
        np.testing.assert_allclose(
            analytic_gradient[name], numerical_gradient[name], rtol=1e-6
        )

    broken = create_item_type(
        "broken",
        logistic,
        gradient_function=lambda theta, a, b: {"a": theta},
        par_names=["a", "b"],
    )
    with pytest.raises(MirtValidationError, match="every named parameter"):
        CustomItemModel(1, broken).parameter_gradient(theta, 0)

    fixed = create_item_type(
        "fixed",
        lambda theta, fixed: np.full(len(theta), 0.5 + 0.0 * fixed),
        par_names=["fixed"],
        par_bounds={"fixed": (1.0, 1.0)},
        par_defaults={"fixed": 1.0},
    )
    np.testing.assert_array_equal(
        CustomItemModel(1, fixed).parameter_gradient(theta, 0)["fixed"],
        np.zeros(3),
    )


@pytest.mark.parametrize(
    ("callback", "n_categories", "match"),
    [
        (lambda theta: np.ones((len(theta), 2)), 2, "returned shape"),
        (lambda theta: np.full(len(theta), 2.0), 2, r"in \[0, 1\]"),
        (lambda theta: np.full((len(theta), 3), 0.2), 3, "sum to 1"),
    ],
)
def test_probability_callback_output_is_validated(
    callback: Callable[[np.ndarray], np.ndarray], n_categories: int, match: str
) -> None:
    model = CustomItemModel(
        1, create_item_type("broken", callback, par_names=[], n_categories=n_categories)
    )
    with pytest.raises(MirtValidationError, match=match):
        model.probability(np.array([0.0, 1.0]))


def test_information_callback_output_is_validated() -> None:
    model = CustomItemModel(
        1,
        create_item_type(
            "broken info",
            lambda theta: np.full(len(theta), 0.5),
            info_function=lambda theta: np.full(len(theta), -1.0),
            par_names=[],
        ),
    )
    with pytest.raises(MirtValidationError, match="non-negative"):
        model.information(np.array([0.0, 1.0]))


@pytest.mark.parametrize(
    "responses",
    [
        np.array([0, 1]),
        np.array([[0], [1]]),
        np.array([[0.5, 1.0]]),
        np.array([[0, 2]]),
        np.array([[0, -2]]),
        np.array([[0.0, np.nan]]),
    ],
)
def test_invalid_responses_are_rejected(
    logistic_spec: ItemTypeSpec, responses: np.ndarray
) -> None:
    model = CustomItemModel(2, logistic_spec)
    with pytest.raises(MirtDataError):
        model.log_likelihood_batch(responses, np.array([0.0]))


def test_invalid_theta_item_and_category_are_rejected(
    logistic_spec: ItemTypeSpec,
) -> None:
    model = CustomItemModel(1, logistic_spec)
    for action in (
        lambda: model.probability(np.array([])),
        lambda: model.probability(np.array([np.nan])),
        lambda: model.probability(np.zeros((2, 2))),
        lambda: model.probability(np.array([0.0]), 2),
        lambda: model.category_probability(np.array([0.0]), 0, 2),
    ):
        with pytest.raises((MirtValidationError, IndexError)):
            action()


def test_callable_shortcut_and_standard_registry() -> None:
    model = CustomItemModel(1, lambda theta: np.full(len(theta), 0.25))
    np.testing.assert_array_equal(
        model.probability(np.array([0.0, 1.0])), [[0.25], [0.25]]
    )
    assert list_standard_item_types() == [
        "STANDARD_2PL",
        "STANDARD_3PL",
        "LOGISTIC_DEVIATION",
    ]
    assert get_standard_item_type("STANDARD_2PL") is STANDARD_2PL
    assert get_standard_item_type("LOGISTIC_DEVIATION") is LOGISTIC_DEVIATION
    with pytest.raises(MirtValidationError, match="Unknown item type"):
        get_standard_item_type("missing")


def test_group_inference_routes_parameters_to_each_callback() -> None:
    spec = create_group(
        " Free ",
        mean_function=lambda mu: mu,
        cov_function=lambda sigma: np.array([[sigma**2]]),
        par_bounds={"mu": (-3.0, 3.0), "sigma": (0.1, 3.0)},
        par_defaults={"mu": 0.25, "sigma": 1.5},
    )
    model = CustomGroupModel(spec)

    assert spec.name == "Free"
    assert spec.par_names == ["mu", "sigma"]
    np.testing.assert_array_equal(model.get_mean(), [0.25])
    np.testing.assert_array_equal(model.get_cov(), [[2.25]])
    assert model.set_parameters(mu=0.5, sigma=2.0) is model
    np.testing.assert_array_equal(model.get_mean(), [0.5])
    np.testing.assert_array_equal(model.get_cov(), [[4.0]])


def test_group_defaults_sampling_and_copy() -> None:
    model = CustomGroupModel(GroupSpec("default", n_factors=2))
    np.testing.assert_array_equal(model.get_mean(), [0.0, 0.0])
    np.testing.assert_array_equal(model.get_cov(), np.eye(2))

    samples = model.sample(4, np.random.default_rng(42))
    assert samples.shape == (4, 2)
    assert model.sample(0, np.random.default_rng(42)).shape == (0, 2)

    copied = model.copy()
    assert copied is not model
    assert copied.parameters == model.parameters


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GroupSpec("", n_factors=1),
        lambda: GroupSpec("x", mean_function=1),  # type: ignore[arg-type]
        lambda: GroupSpec("x", cov_function=1),  # type: ignore[arg-type]
        lambda: GroupSpec("x", n_factors=0),
        lambda: CustomGroupModel("x"),  # type: ignore[arg-type]
    ],
)
def test_invalid_group_specs_fail_early(factory: Callable[[], object]) -> None:
    with pytest.raises(MirtValidationError):
        factory()


@pytest.mark.parametrize(
    "spec, match",
    [
        (create_group("mean", mean_function=lambda: [0.0, 1.0]), "mean_function"),
        (create_group("cov shape", cov_function=lambda: np.eye(2)), "cov_function"),
        (
            create_group(
                "asymmetric", cov_function=lambda: [[1.0, 1.0], [0.0, 1.0]], n_factors=2
            ),
            "symmetric",
        ),
        (
            create_group(
                "indefinite", cov_function=lambda: [[1.0, 2.0], [2.0, 1.0]], n_factors=2
            ),
            "positive-semidefinite",
        ),
    ],
)
def test_group_callback_output_is_validated(spec: GroupSpec, match: str) -> None:
    model = CustomGroupModel(spec)
    with pytest.raises(MirtValidationError, match=match):
        model.get_mean() if spec.mean_function is not None else model.get_cov()


def test_group_parameter_and_sampling_validation() -> None:
    model = CustomGroupModel(
        create_group(
            "bounded",
            par_names=["mu"],
            par_bounds={"mu": (-1.0, 1.0)},
            par_defaults={"mu": 0.0},
        )
    )
    for action in (
        lambda: model.set_parameters(nope=0.0),
        lambda: model.set_parameters(mu=2.0),
        lambda: model.set_parameters(mu=np.array([0.0])),
        lambda: model.sample(-1),
        lambda: model.sample(1, np.random.RandomState(1)),
    ):
        with pytest.raises(MirtValidationError):
            action()


def test_group_aliases_and_public_exports() -> None:
    assert createGroup("compat") == create_group("compat")
    assert mirt.create_group is create_group
    assert mirt.createGroup is createGroup
    assert mirt.CustomGroupModel is CustomGroupModel
    assert mirt.CustomItemModel is CustomItemModel
