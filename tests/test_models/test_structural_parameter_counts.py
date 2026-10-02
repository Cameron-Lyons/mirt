"""Count declared coefficients and verify their statistical parameterization."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit, logsumexp, softmax

from mirt.diagnostics import modelfit
from mirt.estimation.em import EMEstimator
from mirt.exceptions import MirtValidationError
from mirt.models.cdm import DINA, DINO
from mirt.models.cdm_advanced import GDINA
from mirt.models.custom import CustomItemModel, create_item_type
from mirt.models.explanatory import LLTM, ExplanatoryIRT, RaschLLTM
from mirt.models.nested import FourPLNestedLogit, ThreePLNestedLogit, TwoPLNestedLogit
from mirt.models.polytomous import (
    GradedRatingScaleModel,
    NominalResponseModel,
    RatingScaleModel,
)
from mirt.models.sequential import (
    AdjacentCategoryModel,
    ContinuationRatioModel,
    SequentialResponseModel,
)
from mirt.models.testlet import (
    BifactorTestletModel,
    RandomTestletEffectsModel,
    TestletModel,
)


def test_gdina_counts_declared_reduced_designs_not_metadata_or_padding():
    q_matrix = np.array(
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    )
    model = GDINA(
        6,
        3,
        q_matrix,
        reduced_models=["saturated", "saturated", "DINA", "ACDM", "LLM", "RRUM"],
    )
    # 2**1 + 2**2 + 2 + (3+1) + (3+1) + (3+1) independently from the design.
    assert model.n_parameters == 20
    masks = model.free_parameter_masks
    assert_array_equal(masks["delta"].sum(axis=1), [2, 4, 2, 4, 4, 4])
    assert not masks["delta_n_params"].any()
    assert not masks["reduced_model_code"].any()
    patterns = model.attribute_patterns
    original = model.probability(patterns)
    storage = model.parameters["delta"]
    storage[0, 2:] = [100.0, -100.0]
    storage[2, 2:] = [100.0, -100.0]
    model.set_parameters(delta=storage)
    assert_allclose(model.probability(patterns), original, atol=0, rtol=0)

    restricted = model.free_parameter_masks["delta"]
    restricted[1, 2] = False
    model.set_free_parameter_masks({"delta": restricted})
    copied = model.copy()
    assert model.n_parameters == copied.n_parameters == 19
    copied.set_free_parameter_masks(None)
    assert copied.n_parameters == 20
    assert model.n_parameters == 19
    for name in ("delta", "delta_n_params", "reduced_model_code"):
        invalid = model.free_parameter_masks[name]
        invalid.flat[-1 if name != "delta" else 2] = True
        if name == "delta":
            invalid[0, 2] = True
        with pytest.raises(MirtValidationError, match="model-family fixed"):
            model.set_free_parameter_masks({name: invalid})


@pytest.mark.parametrize("family", [DINA, DINO])
def test_cdm_without_required_attributes_has_one_response_probability(family):
    model = family(2, 2, np.array([[0, 0], [1, 1]]))
    assert model.n_parameters == 3
    assert_array_equal(model.free_parameter_masks["guess"], [False, True])
    original = model.probability(model.attribute_patterns)
    model.set_parameters(guess=np.array([0.7, 0.2]))
    assert_array_equal(model.probability(model.attribute_patterns), original)


@pytest.mark.parametrize(
    "reduced", ["saturated", "DINA", "DINO", "ACDM", "LLM", "RRUM"]
)
def test_gdina_empty_required_design_has_one_coefficient(reduced):
    model = GDINA(1, 2, np.zeros((1, 2), dtype=int), reduced_models=[reduced])
    assert model.n_parameters == 1
    original = model.probability(model.attribute_patterns)
    if reduced in {"DINA", "DINO"}:
        values = model.parameters["delta"]
        unused = 0 if reduced == "DINA" else 1
        assert not model.free_parameter_masks["delta"][0, unused]
        values[0, unused] += 0.1
        model.set_parameters(delta=values)
        assert_array_equal(model.probability(model.attribute_patterns), original)


@pytest.mark.parametrize("family", [LLTM, RaschLLTM, ExplanatoryIRT])
def test_explanatory_counts_fixed_shared_and_unused_design_coefficients(family):
    features = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    kwargs = {"n_person_covariates": 1} if family is ExplanatoryIRT else {}
    model = family(3, features, **kwargs)
    assert model.n_parameters == (2 if family is RaschLLTM else 3)
    assert_array_equal(
        model.free_parameter_masks["feature_weights"], [True, True, False]
    )
    theta = np.array([[-1.0], [0.0], [1.0]])
    original = model.probability(theta)
    model.set_feature_weights(np.array([0.0, 0.0, 9.0]))
    assert_array_equal(model.probability(theta), original)
    if family is RaschLLTM:
        assert not model.free_parameter_masks["discrimination"].any()
        with pytest.raises(ValueError, match="fixed to 1"):
            model.set_parameters(discrimination=np.full(3, 1.2))
    else:
        assert_array_equal(
            model.free_parameter_masks["discrimination"], [True, False, False]
        )
        with pytest.raises(ValueError, match="common"):
            model.set_parameters(discrimination=np.array([1.0, 1.2, 1.0]))
        canonical = model._canonical_parameter_values(
            "discrimination", np.array([1.2, 1.0, 1.0])
        )
        assert_array_equal(canonical, [1.2, 1.2, 1.2])


@pytest.mark.parametrize("family", [LLTM, ExplanatoryIRT])
def test_unconstrained_explanatory_slope_counts(family):
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    kwargs = {"n_person_covariates": 1} if family is ExplanatoryIRT else {}
    model = family(3, features, constrain_discrimination=False, **kwargs)
    assert model.n_parameters == 2 + 3
    assert model.free_parameter_masks["discrimination"].all()


@pytest.mark.parametrize(
    "family", [SequentialResponseModel, ContinuationRatioModel, AdjacentCategoryModel]
)
def test_ordinal_process_padding_is_not_estimated(family):
    model = family(3, [2, 4, 3])
    assert model.n_parameters == 3 + 1 + 3 + 2
    assert_array_equal(model.free_parameter_masks["thresholds"].sum(axis=1), [1, 3, 2])
    theta = np.linspace(-2, 2, 7)[:, None]
    original = model.probability(theta)
    values = model.parameters["thresholds"]
    values[0, 1:] = [100.0, -100.0]
    values[2, 2] = 100.0
    model.set_parameters(thresholds=values)
    assert_array_equal(model.probability(theta), original)
    masks = model.free_parameter_masks
    masks["thresholds"][0, 1] = True
    with pytest.raises(MirtValidationError, match="model-family fixed"):
        model.set_free_parameter_masks(masks)


def test_custom_equal_bounds_are_fixed_coefficients():
    spec = create_item_type(
        "KnownSlope",
        lambda theta, a, b: expit(a * (theta - b)),
        par_defaults={"a": 1.5, "b": 0.0},
        par_bounds={"a": (1.5, 1.5), "b": (-3.0, 3.0)},
    )
    model = CustomItemModel(3, spec)
    assert model.n_parameters == 3
    assert not model.free_parameter_masks["a"].any()
    with pytest.raises(MirtValidationError):
        model.set_parameters(a=np.full(3, 1.6))
    masks = model.free_parameter_masks
    masks["a"][0] = True
    with pytest.raises(MirtValidationError, match="model-family fixed"):
        model.set_free_parameter_masks(masks)


@pytest.mark.parametrize(
    "family", [TestletModel, BifactorTestletModel, RandomTestletEffectsModel]
)
def test_standalone_testlet_loading_is_fixed(family):
    model = family(3, [0, 0, -1])
    assert model.n_parameters == 3 + 2 + 3 + 1
    assert_array_equal(
        model.free_parameter_masks["testlet_loadings"], [True, True, False]
    )
    with pytest.raises(MirtValidationError, match="standalone"):
        model.set_parameters(testlet_loadings=np.array([0.5, 0.5, 0.1]))


@pytest.mark.parametrize(
    "family,extra", [(RatingScaleModel, 0), (GradedRatingScaleModel, 1)]
)
def test_rating_scale_canonical_origin_matches_raw_formula(family, extra):
    model = family(3, 4)
    theta = np.linspace(-2.0, 2.0, 13)[:, None]
    difficulty = np.array([-0.4, 0.2, 1.1])
    thresholds = np.array([-1.3, 0.2, 1.0])
    kwargs = {"discrimination": 1.4} if extra else {}
    increments = (
        theta[:, :, None] - difficulty[None, :, None] - thresholds[None, None, :]
    )
    if extra:
        cumulative = expit(1.4 * increments)
        reference = -np.diff(
            np.concatenate(
                (np.ones((13, 3, 1)), cumulative, np.zeros((13, 3, 1))), axis=2
            ),
            axis=2,
        )
    else:
        reference = softmax(
            np.concatenate(
                (np.zeros((13, 3, 1)), np.cumsum(increments, axis=2)), axis=2
            ),
            axis=2,
        )
    model.set_parameters(difficulty=difficulty, thresholds=thresholds, **kwargs)
    assert_allclose(model.probability(theta), reference, atol=2e-15)
    assert model.thresholds[0] == 0.0
    assert_array_equal(model.difficulty, difficulty + thresholds[0])
    assert model.n_parameters == 3 + (4 - 2) + extra

    # A different raw origin represents exactly the same curves.
    model.set_parameters(difficulty=difficulty + 3, thresholds=thresholds - 3, **kwargs)
    assert_allclose(model.probability(theta), reference, atol=2e-15)
    mask = model.free_parameter_masks["thresholds"]
    mask[0] = True
    with pytest.raises(MirtValidationError, match="model-family fixed"):
        model.set_free_parameter_masks({"thresholds": mask})
    before = model.parameters
    with pytest.raises(ValueError):
        model.set_parameters(difficulty=np.ones(3), thresholds=np.ones(2), **kwargs)
    for name, values in before.items():
        assert_array_equal(model.parameters[name], values)
    model.set_item_parameter(1, "difficulty", 0.7)
    assert model.difficulty[1] == 0.7
    assert model.thresholds[0] == 0.0
    with pytest.raises(MirtValidationError, match="shared"):
        model.set_item_parameter(0, "thresholds", 0.1)


@pytest.mark.parametrize(
    "family,per_item",
    [(TwoPLNestedLogit, 2), (ThreePLNestedLogit, 3), (FourPLNestedLogit, 4)],
)
def test_nested_reference_contrasts_match_arbitrary_raw_distractors(family, per_item):
    counts = [2, 4, 3]
    keys = [1, 2, 0]
    model = family(3, counts, correct_response=keys)
    theta = np.linspace(-2, 2, 11)[:, None]
    slopes = np.array(
        [[3.0, -1.0, 8.0, -9.0], [-1.0, 0.4, 6.0, 1.0], [5.0, 0.1, -0.8, 7.0]]
    )
    intercepts = np.array(
        [[0.4, 2.0, 8.0, -9.0], [0.8, -0.2, 6.0, 0.5], [5.0, 0.4, -0.3, 7.0]]
    )
    reference = np.zeros((11, 3, 4))
    correct_prob = model.probability(theta)
    for item, (count, key) in enumerate(zip(counts, keys, strict=True)):
        distractors = np.delete(np.arange(count), key)
        conditional = softmax(
            theta * slopes[item, distractors] + intercepts[item, distractors], axis=1
        )
        p_correct = correct_prob[:, item, key]
        reference[:, item, distractors] = (1 - p_correct[:, None]) * conditional
        reference[:, item, key] = p_correct
    model.set_parameters(distractor_slopes=slopes, distractor_intercepts=intercepts)
    assert_allclose(model.probability(theta), reference, atol=2e-15)
    model.set_parameters(
        distractor_slopes=slopes + 3, distractor_intercepts=intercepts - 2
    )
    assert_allclose(model.probability(theta), reference, atol=2e-15)
    assert model.n_parameters == 3 * per_item + 2 * ((2 - 2) + (4 - 2) + (3 - 2))
    for item, key in enumerate(keys):
        baseline = 0 if key != 0 else 1
        for name in ("distractor_slopes", "distractor_intercepts"):
            assert model.parameters[name][item, baseline] == 0.0
            assert model.parameters[name][item, key] == 0.0
            assert not model.free_parameter_masks[name][item, baseline]
    mask = model.free_parameter_masks["distractor_slopes"]
    mask[1, 0] = True
    with pytest.raises(MirtValidationError, match="model-family fixed"):
        model.set_free_parameter_masks({"distractor_slopes": mask})
    before = model.parameters
    with pytest.raises(MirtValidationError):
        model.set_parameters(distractor_slopes=slopes, difficulty=np.ones(2))
    for name, values in before.items():
        assert_array_equal(model.parameters[name], values)
    model.set_item_parameter(1, "distractor_slopes", slopes[1])
    assert model.distractor_slopes[1, 0] == 0.0
    assert_allclose(model.probability(theta), reference, atol=2e-15)


@pytest.mark.parametrize("n_factors", [1, 2])
def test_nominal_public_setters_preserve_raw_probabilities_and_reference_zero(
    n_factors,
):
    model = NominalResponseModel(2, [2, 4], n_factors)
    theta = np.arange(6 * n_factors).reshape(6, n_factors) / 5 - 1
    shape = (2, 4) if n_factors == 1 else (2, 4, n_factors)
    slopes = np.arange(np.prod(shape), dtype=float).reshape(shape) / 10 + 0.2
    intercepts = np.array([[0.4, -0.3, 100.0, -100.0], [0.7, 0.2, -0.6, 0.4]])
    reference = np.zeros((6, 2, 4))
    for item, count in enumerate([2, 4]):
        logits = (
            theta[:, 0, None] * slopes[item, :count]
            if n_factors == 1
            else theta @ slopes[item, :count].T
        )
        reference[:, item, :count] = softmax(logits + intercepts[item, :count], axis=1)
    model.set_parameters(slopes=slopes, intercepts=intercepts)
    assert_allclose(model.probability(theta), reference, atol=2e-15)
    model.set_parameters(slopes=slopes + 1, intercepts=intercepts - 2)
    assert_allclose(model.probability(theta), reference, atol=2e-15)
    assert model.n_parameters == ((2 - 1) + (4 - 1)) * (n_factors + 1)
    assert_array_equal(model.slopes[:, 0], 0)
    assert_array_equal(model.intercepts[:, 0], 0)
    before = model.parameters
    with pytest.raises(MirtValidationError):
        model.set_parameters(slopes=slopes + 1, intercepts=np.ones((2, 3)))
    for name, values in before.items():
        assert_array_equal(model.parameters[name], values)
    model.set_item_parameter(1, "slopes", slopes[1])
    assert_allclose(model.probability(theta), reference, atol=2e-15)


def test_common_slope_and_global_feature_moment_derivatives_match_analytic_oracle():
    # The global feature array length equals n_items, with a full-rank design
    # whose columns each influence multiple items.
    features = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 2.0]])
    effects = np.array([0.4, -0.3, 0.2])
    slope = 1.3
    model = LLTM(3, features).set_parameters(
        feature_weights=effects, discrimination=np.full(3, slope)
    )
    nodes, masses = np.polynomial.hermite.hermgauss(41)
    nodes *= np.sqrt(2)
    masses /= np.sqrt(np.pi)
    locations = features @ effects
    p = expit(slope * (nodes[:, None] - locations))
    derivatives = np.zeros((41, 3, 4))
    derivatives[:, :, :3] = (
        -slope * p[:, :, None] * (1 - p[:, :, None]) * features[None]
    )
    derivatives[:, :, 3] = p * (1 - p) * (nodes[:, None] - locations)
    pair_derivatives = np.stack(
        [
            derivatives[:, i] * p[:, j, None] + derivatives[:, j] * p[:, i, None]
            for i, j in [(0, 1), (0, 2), (1, 2)]
        ],
        axis=1,
    )
    expected = np.einsum(
        "q,qmp->mp", masses, np.concatenate((derivatives, pair_derivatives), axis=1)
    )
    responses = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1]] * 10)
    actual = modelfit._model_moment_jacobian(
        model, responses, None, 41, modelfit._moment_design(responses)
    )
    assert model.n_parameters == 4
    assert actual.shape == (6, 4)
    assert_allclose(actual, expected, atol=3e-10, rtol=2e-8)
    result = modelfit.compute_m2(model, responses, n_quadpts=41)
    assert result["df"] == 2


def test_declared_equal_testlet_loadings_count_one_coefficient_per_testlet():
    model = BifactorTestletModel(5, [3, 3, 8, 8, -1], constrain_testlet_loadings=True)
    assert model.n_parameters == 5 + 5 + 2 + 2
    assert_array_equal(
        model.free_parameter_masks["testlet_loadings"],
        [True, False, True, False, False],
    )
    before = model.parameters
    with pytest.raises(MirtValidationError, match="common value"):
        model.set_parameters(
            difficulty=np.ones(5), testlet_loadings=np.array([0.5, 0.6, 0.7, 0.7, 0.0])
        )
    for name, values in before.items():
        assert_array_equal(model.parameters[name], values)
    canonical = model._canonical_parameter_values(
        "testlet_loadings", np.array([0.8, 0.5, 0.9, 0.7, 0.0])
    )
    assert_array_equal(canonical, [0.8, 0.8, 0.9, 0.9, 0.0])
    model.set_parameters(testlet_loadings=canonical)
    model.set_testlet_loadings(np.array([0.7, 0.9, 1.0, 0.8, 0.0]))
    assert_allclose(model.testlet_loadings, [0.8, 0.8, 0.9, 0.9, 0.0])


@pytest.mark.parametrize("family", ["RSM", "GRSM", "nested", "nominal"])
def test_canonical_family_copy_retains_additional_fixed_coefficient(family):
    factories = {
        "RSM": lambda: RatingScaleModel(3, 4),
        "GRSM": lambda: GradedRatingScaleModel(3, 4),
        "nested": lambda: TwoPLNestedLogit(3, [2, 4, 3]),
        "nominal": lambda: NominalResponseModel(3, [2, 4, 3]),
    }
    model = factories[family]()
    name = "intercepts" if family == "nominal" else "difficulty"
    original_count = model.n_parameters
    mask = model.free_parameter_masks[name]
    index = 1 if family == "nominal" else 0
    assert mask.flat[index]
    mask.flat[index] = False
    model.set_free_parameter_masks({name: mask})
    copied = model.copy()
    assert model.n_parameters == copied.n_parameters == original_count - 1
    copied.set_free_parameter_masks(None)
    assert copied.n_parameters == original_count
    assert model.n_parameters == original_count - 1


def test_actual_ordinal_fit_criteria_use_active_threshold_counts():
    truth = AdjacentCategoryModel(4, [2, 3, 4, 2])
    theta = np.random.default_rng(227).normal(size=(300, 1))
    probabilities = truth.probability(theta)
    responses = np.sum(
        np.random.default_rng(567).random((300, 4, 1))
        > np.cumsum(probabilities, axis=2),
        axis=2,
    )
    result = EMEstimator(
        n_quadpts=11,
        max_iter=100,
        tol=1e-4,
        use_rust=False,
        use_gpu=False,
        compute_standard_errors=False,
    ).fit(AdjacentCategoryModel(4, [2, 3, 4, 2]), responses)
    assert result.converged
    nodes, masses = np.polynomial.hermite.hermgauss(11)
    curves = result.model.probability((nodes * np.sqrt(2))[:, None])
    terms = sum(np.log(curves[:, item, responses[:, item]]).T for item in range(4))
    direct_likelihood = float(
        np.sum(logsumexp(terms + np.log(masses / np.sqrt(np.pi)), axis=1))
    )
    assert_allclose(result.log_likelihood, direct_likelihood, atol=1e-8)
    assert result.n_parameters == 4 + (1 + 2 + 3 + 1)
    assert_allclose(result.aic, -2 * direct_likelihood + 22)
    assert_allclose(result.bic, -2 * direct_likelihood + 11 * np.log(300))


@pytest.mark.parametrize("family", [LLTM, ExplanatoryIRT])
@pytest.mark.parametrize("independent_error", [0.2, 0.0, np.nan, np.inf])
def test_common_discrimination_uncertainty_follows_linear_copy_map(
    family, independent_error
):
    features = np.eye(3)
    kwargs = {"n_person_covariates": 1} if family is ExplanatoryIRT else {}
    model = family(3, features, **kwargs)
    errors = np.array([independent_error, 0.0, 0.0])
    original = errors.copy()
    actual = model._expand_parameter_standard_errors("discrimination", errors)
    # The common coefficient map a -> (a,a,a) has Jacobian (1,1,1).
    # Thus every copy has the independent coefficient's variance, including
    # unknown, unbounded, and explicitly known coefficient uncertainty.
    assert_array_equal(actual, np.full(3, independent_error))
    assert_array_equal(errors, original)
    assert not np.shares_memory(actual, errors)
    ordinary = np.array([0.1, 0.2, 0.3])
    assert_array_equal(
        model._expand_parameter_standard_errors("feature_weights", ordinary), ordinary
    )


def test_unconstrained_discrimination_uncertainty_is_not_broadcast():
    model = LLTM(3, np.eye(3), constrain_discrimination=False)
    errors = np.array([0.1, 0.2, 0.3])
    assert_array_equal(
        model._expand_parameter_standard_errors("discrimination", errors), errors
    )


@pytest.mark.parametrize("first", [0.3, 0.0, np.nan, np.inf])
def test_common_testlet_loading_uncertainty_follows_grouped_linear_map(first):
    model = BifactorTestletModel(5, [3, 3, 8, 8, -1], constrain_testlet_loadings=True)
    errors = np.array([first, 0.0, 0.2, 0.0, 0.0])
    original = errors.copy()
    actual = model._expand_parameter_standard_errors("testlet_loadings", errors)
    assert_array_equal(actual, [first, first, 0.2, 0.2, 0.0])
    assert_array_equal(errors, original)
    assert not np.shares_memory(actual, errors)
    ordinary = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert_array_equal(
        model._expand_parameter_standard_errors("difficulty", ordinary), ordinary
    )
