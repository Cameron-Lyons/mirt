"""EM calibration of mixed-format tests and its downstream consumers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import mirt
import mirt.estimation._louis_information as louis
from mirt.diagnostics.itemfit import _sx2_parameter_counts
from mirt.estimation.base import _parameter_bounds
from mirt.estimation.em import EMEstimator
from mirt.estimation.mcem import MCEMEstimator
from mirt.estimation.mixed_format_em import MixedFormatEMEstimator, em_estimator_for
from mirt.estimation.priors import (
    BetaPrior,
    LogNormalPrior,
    NormalPrior,
    PriorSpecification,
)
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.standard_errors import (
    _finite_difference_information,
    _finite_difference_scores,
    _flatten_parameters,
)
from mirt.exceptions import MirtModelError, MirtValidationError
from mirt.models.dichotomous import ThreeParameterLogistic, TwoParameterLogistic
from mirt.models.mixed_format import MixedItemModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedRatingScaleModel,
    GradedResponseModel,
    NominalResponseModel,
    RatingScaleModel,
)
from mirt.results.fit_result import FitResult
from mirt.utils.calibration import fixed_item_calibration
from mirt.utils.starting import gen_random_pars, multi_start_fit

TYPES = ["3PL"] * 6 + ["GRM"] * 3 + ["2PL"] * 3


def _true_model(types: list[str], rng: np.random.Generator) -> MixedItemModel:
    model = MixedItemModel.from_itemtypes(types, n_categories=4)
    values = {}
    for name, current in model.parameters.items():
        family, parameter = name.split(".")
        size = current.shape[0]
        if parameter == "discrimination":
            values[name] = rng.uniform(0.9, 2.0, current.shape)
        elif parameter == "difficulty":
            values[name] = rng.normal(0.0, 0.9, size)
        elif parameter == "guessing":
            values[name] = rng.uniform(0.1, 0.25, size)
        elif parameter == "thresholds":
            # Separated thresholds keep every category observed.
            gaps = rng.uniform(0.6, 1.4, current.shape)
            values[name] = np.cumsum(gaps, axis=1) - gaps.sum(axis=1)[:, None] / 2
    model.set_parameters(**values)
    model._is_fitted = True
    return model


@pytest.fixture(scope="module")
def data() -> tuple[MixedItemModel, np.ndarray]:
    rng = np.random.default_rng(42)
    true = _true_model(TYPES, rng)
    responses = true.simulate(rng.standard_normal((800, 1)), seed=43)
    responses[rng.random(responses.shape) < 0.04] = -1
    return true, responses


@pytest.fixture(scope="module")
def fitted(data) -> FitResult:
    _, responses = data
    return mirt.fit_mirt(responses, model=TYPES, tol=1e-3)


def _items_of(model: MixedItemModel, family: str) -> list[int]:
    return [item for item, name in enumerate(model.item_types) if name == family]


@pytest.mark.parametrize(
    ("component", "options"),
    [
        (lambda: GradedResponseModel(4, n_categories=[3, 4, 4, 5]), {}),
        (
            lambda: GradedResponseModel(4, n_categories=[3, 4, 4, 5]),
            {"se_method": "complete_data"},
        ),
        (lambda: GeneralizedPartialCredit(4, n_categories=4), {}),
        (lambda: TwoParameterLogistic(4), {}),
        # Without native kernels the 3PL uses the itemwise path in both fits.
        (lambda: ThreeParameterLogistic(4), {"use_rust": False}),
    ],
)
def test_one_component_reproduces_component_fit(component, options) -> None:
    rng = np.random.default_rng(11)
    model = component()
    responses = (
        _true_model(["GRM"] * 4, rng).simulate(rng.standard_normal((400, 1)), seed=1)
        if model.is_polytomous
        else (rng.random((400, 4)) < 0.6).astype(int)
    )
    if model.is_polytomous:
        responses = np.minimum(responses, np.asarray(model.n_categories) - 1)
    responses[rng.random(responses.shape) < 0.05] = -1

    reference = EMEstimator(**options).fit(model.copy(), responses)
    mixed = MixedItemModel([(model.copy(), range(4))])
    result = MixedFormatEMEstimator(**options).fit(mixed, responses)

    assert result.n_iterations == reference.n_iterations
    assert result.log_likelihood == pytest.approx(reference.log_likelihood, abs=1e-8)
    assert result.n_parameters == reference.n_parameters
    assert result.se_method == reference.se_method
    prefix = mixed.component_names[0]
    for name, values in reference.model.parameters.items():
        np.testing.assert_allclose(
            result.model.parameters[f"{prefix}.{name}"], values, atol=1e-8
        )
        np.testing.assert_allclose(
            result.standard_errors[f"{prefix}.{name}"],
            reference.standard_errors[name],
            atol=1e-8,
        )
    if reference.vcov is None:
        assert result.vcov is None
    else:
        np.testing.assert_allclose(result.vcov, reference.vcov, atol=1e-10)


def test_recovers_three_pl_and_graded_parameters() -> None:
    rng = np.random.default_rng(2026)
    types = ["3PL"] * 20 + ["GRM"] * 5
    true = _true_model(types, rng)
    responses = true.simulate(rng.standard_normal((3000, 1)), seed=7)

    result = mirt.fit_mirt(responses, model=types, tol=1e-3)
    model = result.model

    assert isinstance(model, MixedItemModel)
    assert result.converged
    assert result.se_method == "oakes"
    assert result.n_parameters == 20 * 3 + 5 * 4
    mc, cr = _items_of(model, "3PL"), _items_of(model, "GRM")
    estimate, target = model.item_parameter_arrays(), true.item_parameter_arrays()
    assert (
        np.corrcoef(estimate["difficulty"][mc], target["difficulty"][mc])[0, 1] > 0.97
    )
    assert (
        np.corrcoef(estimate["discrimination"][mc], target["discrimination"][mc])[0, 1]
        > 0.8
    )
    assert np.mean(np.abs(estimate["guessing"][mc] - target["guessing"][mc])) < 0.08
    np.testing.assert_allclose(
        estimate["discrimination"][cr], target["discrimination"][cr], atol=0.2
    )
    np.testing.assert_allclose(
        estimate["thresholds"][cr], target["thresholds"][cr], atol=0.2
    )
    errors = result.standard_errors
    assert np.all(np.isfinite(errors["GRM.thresholds"]))
    assert np.all(errors["GRM.thresholds"] < 0.15)


@pytest.mark.parametrize("reordered", [False, True])
def test_mixed_louis_information_matches_marginal_differences(reordered) -> None:
    three_pl = ThreeParameterLogistic(2)
    three_pl.set_parameters(
        discrimination=np.array([1.3, 0.9]),
        difficulty=np.array([-0.2, 0.6]),
        guessing=np.array([0.15, 0.2]),
    )
    graded = GradedResponseModel(2, n_categories=4)
    graded.set_parameters(
        discrimination=np.array([1.1, 1.6]),
        thresholds=np.array([[-1.0, 0.0, 1.2], [-0.5, 0.3, 0.9]]),
    )
    mixed = MixedItemModel([(three_pl, [0, 2]), (graded, [1, 3])])
    if reordered:
        # Regression: columns followed the component's own storage order
        # instead of the mixed-format layout.
        stored = mixed.component_models[1]._parameters
        mixed.component_models[1]._parameters = dict(reversed(stored.items()))
    rng = np.random.default_rng(5)
    responses = mixed.simulate(rng.standard_normal((150, 1)), seed=6)
    responses[rng.random(responses.shape) < 0.1] = -1
    quadrature = GaussHermiteQuadrature(11)
    mass = quadrature.weights / quadrature.weights.sum()
    _, layouts = _flatten_parameters(mixed)
    assert louis.supports_louis_information(mixed)

    terms = louis.louis_information(mixed, responses, quadrature.nodes, mass, layouts)
    reference, _ = _finite_difference_information(
        mixed, responses, quadrature, mass, 1e-4
    )
    scores, _ = _finite_difference_scores(mixed, responses, quadrature, mass, 1e-5)

    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(terms.information, reference, rtol=0, atol=2e-6 * scale)
    np.testing.assert_allclose(
        terms.score_crossproduct,
        scores.T @ scores,
        rtol=0,
        atol=1e-7 * np.max(np.abs(terms.score_crossproduct)),
    )
    # Components share the trait, so the information is not block diagonal.
    n_first = three_pl.n_parameters
    assert np.max(np.abs(terms.information[:n_first, n_first:])) > 1e-3 * scale


def test_complete_data_errors_are_componentwise(data) -> None:
    _, responses = data
    oakes = mirt.fit_mirt(responses, model=TYPES, tol=1e-3)
    complete = mirt.fit_mirt(
        responses, model=TYPES, tol=1e-3, se_method="complete_data"
    )

    assert complete.se_method == "complete_data"
    assert complete.vcov is None
    for name, errors in oakes.standard_errors.items():
        finite = np.isfinite(errors) & np.isfinite(complete.standard_errors[name])
        # Complete-data information exceeds the observed information.
        assert np.all(complete.standard_errors[name][finite] <= errors[finite] + 1e-9)


def test_fit_result_reports_and_coefficients(fitted) -> None:
    model = fitted.model

    assert fitted.n_parameters == model.n_parameters == 6 * 3 + 3 * 4 + 3 * 2
    assert fitted.vcov.shape == (fitted.n_parameters,) * 2
    # Component rows are labeled by their test items, as in coef().
    assert fitted.vcov_labels[0] == "3PL.discrimination[Item_1]"
    assert "GRM.thresholds[Item_7,0]" in fitted.vcov_labels
    assert "3PL.guessing" in fitted.summary()

    coefficients = fitted._coefficient_columns(include_se=True)
    mc, cr = _items_of(model, "3PL"), _items_of(model, "GRM")
    assert np.all(np.isnan(coefficients["guessing"][cr]))
    np.testing.assert_array_equal(
        coefficients["guessing"][mc], model.parameters["3PL.guessing"]
    )
    np.testing.assert_array_equal(
        coefficients["guessing_se"][mc], fitted.standard_errors["3PL.guessing"]
    )
    assert np.all(np.isnan(coefficients["thresholds_1"][mc]))
    assert fitted.coef() is not None

    payload = fitted.to_dict()
    assert payload["model"]["name"] == "Mixed"
    assert payload["model"]["n_categories"] == model.n_categories
    with pytest.raises(MirtValidationError, match="cannot rebuild model 'Mixed'"):
        FitResult.from_dict(payload)


def _identity_vcov_result(
    model: MixedItemModel, labels: list[str] | None = None
) -> FitResult:
    return FitResult(
        model=model,
        log_likelihood=-1.0,
        n_iterations=1,
        converged=True,
        standard_errors={},
        aic=0.0,
        bic=0.0,
        vcov=np.eye(model.n_parameters),
        vcov_labels=labels,
    )


def test_result_labels_follow_component_items() -> None:
    # Regression: rows were labeled by their position within the component.
    two_components = MixedItemModel(
        [
            (GradedResponseModel(3, n_categories=3), [4, 0, 2]),
            (TwoParameterLogistic(2), [3, 1]),
        ],
        item_names=list("abcde"),
    )
    permuted = MixedItemModel(
        [(GradedResponseModel(3, n_categories=3), [2, 0, 1])], item_names=list("xyz")
    )

    result = _identity_vcov_result(two_components)
    assert result.vcov_labels[:2] == ["GRM.discrimination[e]", "GRM.discrimination[a]"]
    assert result.vcov_labels[-1] == "2PL.difficulty[b]"
    assert result._parameter_label("GRM.thresholds", (3, 2), (1, 0)) == "a[0]"
    assert _identity_vcov_result(two_components, result.vcov_labels).vcov is not None

    result = _identity_vcov_result(permuted)
    assert result.vcov_labels[:3] == [
        "GRM.discrimination[z]",
        "GRM.discrimination[x]",
        "GRM.discrimination[y]",
    ]
    assert result._parameter_label("GRM.discrimination", (3,), (0,)) == "z"


def test_fit_mirt_per_item_families_validation(data) -> None:
    _, responses = data

    with pytest.raises(MirtValidationError, match="11 items but the data have 12"):
        mirt.fit_mirt(responses, model=TYPES[:-1])
    with pytest.raises(MirtModelError, match="Unknown model"):
        mirt.fit_mirt(object(), model=["3PL", "Bogus"])
    with pytest.raises(MirtValidationError, match="estimation='EM'"):
        mirt.fit_mirt(responses, model=TYPES, estimation="MHRM")
    with pytest.raises(MirtModelError, match="multidimensional"):
        mirt.fit_mirt(responses, model=TYPES, n_factors=2)
    with pytest.raises(MirtValidationError, match="dichotomous items"):
        mirt.fit_mirt(responses, model=TYPES, n_categories=[4] * 12)

    single = mirt.fit_mirt(responses[:, 6:9], model=["GRM"] * 3, tol=1e-3)
    assert isinstance(single.model, GradedResponseModel)


def test_fit_mirt_start_values_fixed_and_item_names(data) -> None:
    _, responses = data
    names = [f"Q{item}" for item in range(12)]
    guessing = np.full(6, 0.12)

    result = mirt.fit_mirt(
        responses,
        model=TYPES,
        tol=1e-3,
        item_names=names,
        start_values={"3PL.guessing": guessing},
        fixed={"3PL.guessing": True},
    )

    model = result.model
    assert model.item_names == names
    assert model.component_models[0].item_names == names[:6]
    np.testing.assert_array_equal(model.parameters["3PL.guessing"], guessing)
    assert result.n_parameters == 6 * 2 + 3 * 4 + 3 * 2
    assert np.all(
        np.isnan(result.standard_errors["3PL.guessing"])
        | (result.standard_errors["3PL.guessing"] == 0.0)
    )


def test_item_priors_resolve_per_component(data) -> None:
    responses = data[1][:400]
    model = MixedItemModel.from_itemtypes(TYPES, n_categories=4)

    plain = MixedFormatEMEstimator(tol=1e-2).fit(model.copy(), responses)
    qualified = MixedFormatEMEstimator(
        tol=1e-2, item_priors={"3PL.guessing": BetaPrior(5, 17)}
    ).fit(model.copy(), responses)
    shared = MixedFormatEMEstimator(
        tol=1e-2, item_priors={"discrimination": LogNormalPrior(0.0, 0.5)}
    ).fit(model.copy(), responses)
    specification = MixedFormatEMEstimator(
        tol=1e-2, item_priors=PriorSpecification(guessing=BetaPrior(5, 17))
    ).fit(model.copy(), responses)

    assert plain.log_posterior is None
    for result in (qualified, shared, specification):
        assert result.log_posterior is not None
    # A specification applies its default slope and location priors too.
    mapping = MixedFormatEMEstimator(
        tol=1e-2,
        item_priors={
            "discrimination": LogNormalPrior(0.0, 0.5),
            "difficulty": NormalPrior(0.0, 2.0),
            "guessing": BetaPrior(5, 17),
        },
    ).fit(model.copy(), responses)
    assert specification.log_posterior == pytest.approx(mapping.log_posterior)
    for name, values in mapping.model.parameters.items():
        np.testing.assert_allclose(specification.model.parameters[name], values)
    log_prior = sum(
        np.sum(LogNormalPrior(0.0, 0.5).log_pdf(values))
        for name, values in shared.model.parameters.items()
        if name.endswith("discrimination")
    )
    assert shared.log_posterior - shared.log_likelihood == pytest.approx(log_prior)
    assert not np.allclose(
        qualified.model.parameters["3PL.guessing"],
        plain.model.parameters["3PL.guessing"],
    )

    with pytest.raises(MirtValidationError, match="no component"):
        MixedFormatEMEstimator(item_priors={"slipping": BetaPrior(2, 2)}).fit(
            model.copy(), responses
        )
    with pytest.raises(MirtValidationError, match="Unknown parameter"):
        MixedFormatEMEstimator(item_priors={"GRM.guessing": BetaPrior(2, 2)}).fit(
            model.copy(), responses
        )


def test_parallel_m_step_and_missing_responses_match_serial(data) -> None:
    _, responses = data
    model = MixedItemModel.from_itemtypes(TYPES, n_categories=4)

    serial = MixedFormatEMEstimator(tol=1e-3).fit(model.copy(), responses)
    parallel = MixedFormatEMEstimator(tol=1e-3, n_jobs=2).fit(model.copy(), responses)

    assert parallel.log_likelihood == pytest.approx(serial.log_likelihood, abs=1e-8)
    for name, values in serial.model.parameters.items():
        np.testing.assert_allclose(parallel.model.parameters[name], values, atol=1e-8)


def test_squarem_falls_back_to_plain_em(data) -> None:
    _, responses = data
    model = MixedItemModel.from_itemtypes(TYPES, n_categories=4)
    plain = MixedFormatEMEstimator(tol=1e-3).fit(model.copy(), responses)

    with pytest.warns(UserWarning, match="not available for mixed-format"):
        result = MixedFormatEMEstimator(tol=1e-3, accelerate="squarem").fit(
            model.copy(), responses
        )
    assert result.log_likelihood == pytest.approx(plain.log_likelihood)


def test_multidimensional_and_nominal_components() -> None:
    rng = np.random.default_rng(9)
    binary = TwoParameterLogistic(4, n_factors=2)
    binary.set_parameters(
        discrimination=rng.uniform(0.5, 1.5, (4, 2)), difficulty=rng.normal(size=4)
    )
    graded = GradedResponseModel(2, n_categories=3, n_factors=2)
    true = MixedItemModel([(binary, range(4)), (graded, [4, 5])])
    responses = true.simulate(rng.standard_normal((300, 2)), seed=2)

    result = mirt.fit_mirt(
        responses, model=["2PL"] * 4 + ["GRM"] * 2, n_factors=2, n_quadpts=7, tol=1e-2
    )
    assert result.model.n_factors == 2
    assert result.se_method == "complete_data"
    assert np.all(np.isfinite(result.standard_errors["GRM.thresholds"]))

    nominal = MixedItemModel(
        [(TwoParameterLogistic(4), range(4)), (NominalResponseModel(2, 3), [4, 5])]
    )
    fitted = MixedFormatEMEstimator(tol=1e-2).fit(nominal, responses)
    assert fitted.se_method == "complete_data"
    assert _parameter_bounds(nominal, "NRM.slopes") == (-5.0, 5.0)
    assert _parameter_bounds(nominal, "2PL.discrimination") == (0.1, 5.0)


@pytest.mark.parametrize("family", [RatingScaleModel, GradedRatingScaleModel])
def test_components_with_shared_parameters_estimate_them(family) -> None:
    rng = np.random.default_rng(12)
    binary = TwoParameterLogistic(4)
    binary.set_parameters(
        discrimination=rng.uniform(0.9, 1.8, 4), difficulty=rng.normal(size=4)
    )
    rating = family(5, 4)
    rating.set_parameters(
        difficulty=rng.normal(0.0, 0.5, 5), thresholds=np.array([-1.0, 0.2, 1.1])
    )
    true = MixedItemModel([(binary, range(4)), (rating, range(4, 9))])
    responses = true.simulate(rng.standard_normal((1500, 1)), seed=13)

    model = MixedItemModel(
        [(TwoParameterLogistic(4), range(4)), (family(5, 4), range(4, 9))]
    )
    name = f"{model.component_names[1]}.thresholds"
    start = model.parameters[name]
    result = MixedFormatEMEstimator(tol=1e-8, se_method="complete_data").fit(
        model, responses
    )

    # Each component's M-step ends with the joint step over its shared
    # coordinates; without it the thresholds would keep their defaults.
    estimate = model.parameters[name]
    assert not np.allclose(estimate, start)
    quadrature = GaussHermiteQuadrature(21, 1)
    mass = quadrature.weights / quadrature.weights.sum()
    gradient = []
    for column in np.flatnonzero(model.free_parameter_masks[name]):
        values = []
        for sign in (1.0, -1.0):
            trial = estimate.copy()
            trial[column] += sign * 1e-5
            model._parameters[name] = trial
            values.append(
                louis._posterior_block(
                    model, responses, quadrature.nodes, np.log(mass)
                )[1].sum()
            )
        model._parameters[name] = estimate
        gradient.append((values[0] - values[1]) / 2e-5)
    assert np.max(np.abs(gradient)) < 0.02
    errors = result.standard_errors[name][model.free_parameter_masks[name]]
    assert np.all(np.isfinite(errors) & (errors > 0))


def test_parameter_bounds_follow_components() -> None:
    model = MixedItemModel.from_itemtypes(["3PL", "GRM"], n_categories=3)

    assert _parameter_bounds(model, "3PL.guessing") == (0.0, 0.5)
    assert _parameter_bounds(model, "GRM.thresholds") == (-6.0, 6.0)
    with pytest.raises(MirtValidationError, match="Unknown parameter"):
        _parameter_bounds(model, "guessing")


@pytest.mark.parametrize(
    "estimator",
    [lambda: EMEstimator(max_iter=3), lambda: MCEMEstimator(max_iter=2, n_samples=50)],
)
def test_single_family_estimators_reject_mixed_models(data, estimator) -> None:
    _, responses = data
    model = MixedItemModel.from_itemtypes(TYPES, n_categories=4)
    before = model.parameters

    # Itemwise M-steps would otherwise leave every parameter unchanged.
    with pytest.raises(MirtModelError, match="MixedFormatEMEstimator"):
        estimator().fit(model, responses)
    for name, values in model.parameters.items():
        np.testing.assert_array_equal(values, before[name])


def test_em_estimator_for_selects_by_model() -> None:
    mixed = MixedItemModel.from_itemtypes(["3PL", "GRM"], n_categories=3)

    assert type(em_estimator_for(mixed, tol=1e-2)) is MixedFormatEMEstimator
    assert type(em_estimator_for(TwoParameterLogistic(2))) is EMEstimator


def test_scoring_and_item_fit(fitted, data) -> None:
    _, responses = data
    model = fitted.model

    for method in ("EAP", "MAP", "ML", "WLE", "EAPsum"):
        theta = np.asarray(mirt.fscores(fitted, responses, method=method).theta)
        assert theta.shape[0] == responses.shape[0]
        assert np.all(np.isfinite(theta))

    counts = _sx2_parameter_counts(model, None)
    expected = [{"3PL": 3, "GRM": 4, "2PL": 2}[name] for name in model.item_types]
    np.testing.assert_array_equal(counts, expected)
    complete = responses[np.all(responses >= 0, axis=1)]
    statistics = mirt.diagnostics.compute_itemfit(
        model, complete, statistics=["S_X2", "infit", "outfit"]
    )
    assert np.all(np.isfinite(statistics["S_X2"]))
    assert np.all(np.isfinite(statistics["infit"]))
    assert mirt.personfit(fitted, responses) is not None


def test_bootstrap_refits_mixed_models(fitted, data) -> None:
    _, responses = data

    errors = mirt.bootstrap_se(fitted, responses[:300], n_bootstrap=2, seed=4)

    assert set(errors) == set(fitted.model.parameters)
    assert all(np.all(np.isfinite(values)) for values in errors.values())


def test_single_family_consumers_raise_clear_errors(fitted) -> None:
    model = fitted.model

    with pytest.raises(MirtModelError, match="mod2values requires a single item"):
        mirt.mod2values(model)
    with pytest.raises(MirtModelError, match="mod2values requires a single item"):
        mirt.extract_item(model, 0)
    with pytest.raises(MirtModelError, match="Core linking requires a single item"):
        mirt.equating.link(model, model.copy(), [0, 1, 2], [0, 1, 2])
    with pytest.raises(MirtModelError, match="transform_parameters requires"):
        mirt.equating.transform_parameters(model, 1.2, 0.1)
    with pytest.raises(MirtModelError, match="MultigroupModel requires"):
        mirt.multigroup.MultigroupModel(model, 2)
    # Regression: anchors were not found and the error named no new items.
    with pytest.raises(MirtModelError, match="set_free_parameter_masks"):
        fixed_item_calibration(np.zeros((3, 12), dtype=int), model, [0, 1])

    scores = mirt.estfun(model, np.zeros((3, 12), dtype=int), np.zeros((3, 1)))
    assert scores.shape == (3, model.n_parameters)


def test_no_warning_without_acceleration(data) -> None:
    _, responses = data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MixedFormatEMEstimator(tol=1e-2).fit(
            MixedItemModel.from_itemtypes(TYPES, n_categories=4), responses
        )


def test_random_starts_and_multi_start_fit_use_component_families(data) -> None:
    _, responses = data
    model = MixedItemModel.from_itemtypes(TYPES, n_categories=4)

    # Regression: qualified names hid the families, so values were noise
    # around the current ones and every multi-start refit was rejected.
    starts = gen_random_pars(model, n_sets=2, seed=3, guessing_range=(0.01, 0.02))
    again = gen_random_pars(model, n_sets=2, seed=3, guessing_range=(0.01, 0.02))

    assert set(starts[0]) == set(model.parameters)
    for name, values in starts[1].items():
        np.testing.assert_array_equal(values, again[1][name])
    assert np.all(
        (starts[0]["3PL.guessing"] >= 0.01) & (starts[0]["3PL.guessing"] <= 0.02)
    )
    thresholds = starts[0]["GRM.thresholds"]
    assert np.all(np.diff(thresholds, axis=1) > 0)

    result = multi_start_fit(model, responses[:300], n_starts=2, seed=1, tol=1e-2)
    assert isinstance(result.model, MixedItemModel)
    assert np.isfinite(result.log_likelihood)


def test_replaced_likelihoods_disable_component_shortcuts() -> None:
    rng = np.random.default_rng(8)
    true = _true_model(["2PL"] * 3 + ["GRM"], rng)
    responses = true.simulate(rng.standard_normal((600, 1)), seed=9)
    model = MixedItemModel.from_itemtypes(true.item_types, n_categories=4)
    plain = MixedFormatEMEstimator(tol=1e-2)
    plain.fit(model.copy(), responses)

    custom = model.copy()
    # An instance hook may couple persons, so identical rows must stay apart
    # and the exact information must not assume the component likelihoods.
    custom.log_likelihood_batch = custom.log_likelihood_batch
    estimator = MixedFormatEMEstimator(tol=1e-2)
    result = estimator.fit(custom, responses)

    assert plain._pattern_frequencies is not None
    assert estimator._pattern_frequencies is None
    assert louis.supports_louis_information(model)
    assert not louis.supports_louis_information(custom)
    assert result.se_method == "complete_data"
