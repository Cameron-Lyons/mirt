"""Shared response simulation for models, simdata, and resampling tools."""

import numpy as np
import pytest
from scipy import stats

from mirt import fit_mirt, simdata
from mirt._categorical import draw_item_responses, sample_categorical_tensor
from mirt.exceptions import MirtModelError
from mirt.models.dichotomous import (
    FourParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.nested import TwoPLNestedLogit
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    RatingScaleModel,
)
from mirt.models.sequential import SequentialResponseModel
from mirt.models.unfolding import GeneralizedGradedUnfolding


def _former_bootstrap_simulator(model, theta, rng):
    """The parametric bootstrap's simulator before the shared helper."""
    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    if probabilities.ndim == 2:
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return (rng.random(probabilities.shape) < probabilities).astype(np.int_)
    probabilities = np.maximum(probabilities, 0.0)
    cumulative = np.cumsum(
        probabilities / probabilities.sum(axis=2, keepdims=True), axis=2
    )
    counts = model.n_categories
    for item, n_categories in enumerate(counts):
        cumulative[:, item, n_categories - 1 :] = 1.0
    uniforms = rng.random((theta.shape[0], model.n_items, 1))
    return (uniforms > cumulative).sum(axis=2).astype(np.int_)


def _former_ppc_simulator(model, theta, rng):
    """The posterior predictive simulator before the shared helper."""
    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    if probabilities.ndim == 2:
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return (rng.random(probabilities.shape) < probabilities).astype(np.int64)
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities = probabilities / probabilities.sum(axis=2, keepdims=True)
    cumulative = np.cumsum(probabilities, axis=2)
    draws = rng.random(probabilities.shape[:2])
    replicated = np.sum(draws[..., None] > cumulative, axis=2)
    category_max = np.asarray(model.n_categories, dtype=np.int64)[None, :] - 1
    return np.minimum(replicated, category_max).astype(np.int64)


def _fitted_models():
    rng = np.random.default_rng(3)
    models = [
        TwoParameterLogistic(6).set_parameters(
            discrimination=rng.uniform(0.7, 2.0, 6), difficulty=rng.normal(size=6)
        ),
        ThreeParameterLogistic(4).set_parameters(guessing=np.full(4, 0.2)),
        GradedResponseModel(5, n_categories=[2, 3, 5, 4, 5]),
        GeneralizedPartialCredit(4, n_categories=4),
        NominalResponseModel(3, n_categories=[3, 4, 4]),
    ]
    graded = models[2]
    thresholds = graded.parameters["thresholds"]
    for item, n_categories in enumerate(graded.n_categories):
        thresholds[item, : n_categories - 1] = np.sort(
            rng.normal(size=n_categories - 1)
        )
    graded.set_parameters(thresholds=thresholds)
    return models


@pytest.mark.parametrize("model", _fitted_models(), ids=lambda model: model.model_name)
def test_shared_simulator_reproduces_former_seeded_streams(model):
    theta = np.random.default_rng(9).standard_normal((997, model.n_factors))

    expected = _former_bootstrap_simulator(model, theta, np.random.default_rng(4))
    replicated = _former_ppc_simulator(model, theta, np.random.default_rng(4))
    shared = draw_item_responses(model, theta, np.random.default_rng(4))
    chunked = draw_item_responses(model, theta, np.random.default_rng(4), chunk_size=61)

    np.testing.assert_array_equal(shared, expected)
    np.testing.assert_array_equal(shared, replicated)
    np.testing.assert_array_equal(chunked, expected)
    assert shared.dtype == np.int_


def test_shared_simulator_leaves_the_generator_in_the_same_state_for_any_chunk():
    model = GradedResponseModel(3, n_categories=4)
    theta = np.zeros((50, 1))
    whole = np.random.default_rng(1)
    chunked = np.random.default_rng(1)

    draw_item_responses(model, theta, whole)
    draw_item_responses(model, theta, chunked, chunk_size=7)

    assert whole.random() == chunked.random()


def test_dichotomous_model_simulation_keeps_its_seeded_stream():
    model = FourParameterLogistic(4).set_parameters(
        guessing=np.full(4, 0.1), upper=np.full(4, 0.9)
    )
    theta = np.linspace(-2, 2, 41)
    probabilities = model.probability(theta)
    expected = np.random.default_rng(7).random(probabilities.shape) < probabilities

    np.testing.assert_array_equal(model.simulate(theta, seed=7), expected)


def test_categorical_sampler_never_draws_padded_categories():
    probabilities = np.zeros((2, 2, 4))
    probabilities[:, 0, :2] = 0.5
    probabilities[:, 1] = 0.25
    # Padded mass on item 0 must be absorbed by its last active category.
    probabilities[:, 0, 3] = 0.5
    uniforms = np.array([[0.99, 0.99], [0.1, 0.6]])

    draws = sample_categorical_tensor(probabilities, np.array([2, 4]), uniforms)

    np.testing.assert_array_equal(draws, [[1, 3], [0, 2]])


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.full((1, 1, 2), np.nan), "finite and nonnegative"),
        (np.array([[[-0.5, 1.5]]]), "finite and nonnegative"),
        (np.zeros((1, 1, 2)), "positive mass"),
    ],
)
def test_categorical_sampler_rejects_invalid_probabilities(values, message):
    with pytest.raises(MirtModelError, match=message):
        sample_categorical_tensor(values, np.array([2]), np.zeros((1, 1)))


def test_shared_simulator_rejects_malformed_model_output():
    class BrokenModel:
        n_items = 2
        model_name = "broken"

        def probability(self, theta):
            return np.full((theta.shape[0], 2), 1.5)

    with pytest.raises(MirtModelError, match="within"):
        draw_item_responses(BrokenModel(), np.zeros((3, 1)), np.random.default_rng())


def _polytomous_models():
    rng = np.random.default_rng(21)
    graded = GradedResponseModel(3, n_categories=[3, 4, 5])
    thresholds = graded.parameters["thresholds"]
    for item, n_categories in enumerate(graded.n_categories):
        thresholds[item, : n_categories - 1] = np.sort(
            rng.normal(size=n_categories - 1)
        )
    graded.set_parameters(thresholds=thresholds)
    return [
        graded,
        GeneralizedPartialCredit(3, n_categories=4),
        NominalResponseModel(3, n_categories=4),
        RatingScaleModel(3, n_categories=4),
        TwoPLNestedLogit(3, n_categories=4),
        SequentialResponseModel(3, n_categories=4),
        GeneralizedGradedUnfolding(3, n_categories=4),
    ]


@pytest.mark.parametrize(
    "model", _polytomous_models(), ids=lambda model: type(model).__name__
)
def test_polytomous_simulation_matches_category_probabilities(model):
    theta = np.full((20_000, 1), 0.3)
    expected = model.probability(theta[:1])[0]

    responses = model.simulate(theta, seed=5, chunk_size=4_999)

    assert responses.shape == (20_000, model.n_items)
    for item, n_categories in enumerate(model.n_categories):
        assert responses[:, item].min() >= 0
        assert responses[:, item].max() < n_categories
        observed = np.bincount(responses[:, item], minlength=n_categories)
        probabilities = expected[item, :n_categories] / expected[item].sum()
        chi_square = stats.chisquare(observed, probabilities * theta.shape[0])
        assert chi_square.pvalue > 1e-4


def test_polytomous_simulation_is_identical_for_every_chunk_size():
    model = _polytomous_models()[0]
    theta = np.random.default_rng(2).standard_normal(83)

    expected = model.simulate(theta, seed=13)

    assert expected.dtype == np.int32
    for chunk_size in (1, 7, 83, 1_000):
        np.testing.assert_array_equal(
            model.simulate(theta, seed=13, chunk_size=chunk_size), expected
        )


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_polytomous_simulation_rejects_invalid_chunk_sizes(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        GradedResponseModel(2, n_categories=3).simulate(
            np.zeros(3), chunk_size=chunk_size
        )


@pytest.fixture(scope="module")
def graded_fit():
    responses = simdata("GRM", n_persons=300, n_items=5, n_categories=4, seed=1)
    return fit_mirt(
        responses, model="GRM", n_categories=4, compute_standard_errors=False
    )


class TestSimdataFromModels:
    """simdata draws new samples from fitted model objects."""

    def test_fit_results_and_models_simulate_reproducibly(self, graded_fit):
        first = simdata(graded_fit, n_persons=400, seed=8)
        second = simdata(graded_fit.model, n_persons=400, seed=8)

        rng = np.random.default_rng(8)
        theta = rng.standard_normal((400, 1))
        expected = draw_item_responses(graded_fit.model, theta, rng)

        assert first.shape == (400, 5)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, expected)

    def test_supplied_abilities_match_model_simulation(self, graded_fit):
        theta = np.linspace(-2, 2, 120)

        simulated = simdata(graded_fit, theta=theta, seed=3)

        np.testing.assert_array_equal(
            simulated, graded_fit.model.simulate(theta, seed=3)
        )

    def test_multidimensional_models_draw_every_factor(self):
        model = TwoParameterLogistic(4, n_factors=2)
        rng = np.random.default_rng(6)
        theta = rng.standard_normal((50, 2))

        simulated = simdata(model, n_persons=50, seed=6)

        np.testing.assert_array_equal(simulated, draw_item_responses(model, theta, rng))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"difficulty": np.zeros(5)}, "named simulation models"),
            ({"return_structural_zeros": True}, "named simulation models"),
            ({"n_items": 4}, "n_items must match"),
            ({"n_factors": 2}, "n_factors must match"),
            ({"theta": np.zeros((3, 2))}, "theta must have shape"),
            ({"theta": np.array([np.nan])}, "finite"),
            ({"n_persons": 0}, "n_persons"),
        ],
    )
    def test_rejects_arguments_that_conflict_with_the_model(
        self, graded_fit, kwargs, message
    ):
        with pytest.raises(ValueError, match=message):
            simdata(graded_fit, **kwargs)

    def test_rejects_models_without_item_response_probabilities(self):
        from mirt.models.cdm import DINA

        with pytest.raises(ValueError, match="dichotomous or polytomous"):
            simdata(DINA(n_items=3, n_attributes=2, q_matrix=np.eye(3, 2)))

    def test_item_count_is_inferred_from_supplied_parameters(self):
        responses = simdata(
            "2PL", discrimination=np.ones(30), difficulty=np.zeros(30), seed=1
        )
        steps = simdata("GPCM", n_categories=3, steps=np.zeros((7, 2)), seed=1)

        assert responses.shape == (500, 30)
        assert steps.shape == (500, 7)
        assert simdata("2PL", seed=1).shape == (500, 20)
        with pytest.raises(ValueError, match="discrimination must have shape"):
            simdata("2PL", n_items=20, discrimination=np.ones(30))
