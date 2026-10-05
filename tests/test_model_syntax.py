"""Tests for mirt_model syntax parsing and confirmatory fit_mirt fits."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mirt
from mirt import FitResult, ModelSpec, fit_mirt, mirt_model
from mirt.estimation.latent_density import FactorCovarianceDensity
from mirt.estimation.priors import LogNormalPrior, NormalPrior, PriorSpecification
from mirt.exceptions import MirtModelError, MirtValidationError
from mirt.model_syntax import ParameterEntry

TWO_FACTORS = "F1 = 1-5\nF2 = 6-10\nCOV = F1*F2"


def _simulate_cfa(seed: int, n_persons: int, rho: float = 0.5):
    rng = np.random.default_rng(seed)
    loadings = np.zeros((10, 2))
    loadings[:5, 0] = rng.uniform(1, 2, 5)
    loadings[5:, 1] = rng.uniform(1, 2, 5)
    theta = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n_persons)
    intercepts = rng.normal(0, 1, 10)
    logits = theta @ loadings.T + intercepts
    responses = (rng.random(logits.shape) < 1 / (1 + np.exp(-logits))).astype(int)
    return responses, loadings, intercepts


@pytest.fixture(scope="module")
def cfa_data():
    return _simulate_cfa(seed=0, n_persons=2000)


@pytest.fixture(scope="module")
def cfa_fit(cfa_data):
    responses, _, _ = cfa_data
    return fit_mirt(
        responses, "2PL", spec=TWO_FACTORS, n_quadpts=15, compute_standard_errors=False
    )


@pytest.fixture(scope="module")
def small_data():
    return _simulate_cfa(seed=1, n_persons=500)[0]


# Parsing


def test_parses_factor_lines_with_ranges_and_lists() -> None:
    spec = mirt_model("F1 = 1-3, 5\nF2 = 4, 6:8")

    assert spec.factors == ("F1", "F2")
    assert spec.loadings == ((0, 1, 2, 4), (3, 5, 6, 7))
    assert spec.n_factors == 2
    pattern = spec.loading_pattern()
    assert pattern.shape == (8, 2)
    assert pattern[:, 0].tolist() == [1, 1, 1, 0, 1, 0, 0, 0]
    assert spec.loading_pattern(10).shape == (10, 2)


def test_tolerates_comments_whitespace_and_continued_lines() -> None:
    spec = mirt_model(
        """
        # verbal and quantitative factors
          Verbal   =   1 - 3 ,
                       4        # continued after a trailing comma
        Quant=5-8

        PRIOR = (1-4, a1, lnorm, 0, 0.5),
                (5-8, a2, lnorm, 0, 0.5)
        FIXED = (
            1, a1)
        """
    )

    assert spec.factors == ("Verbal", "Quant")
    assert spec.loadings == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert len(spec.priors) == 2
    assert spec.fixed == (ParameterEntry((0,), ("a1",)),)


def test_resolves_item_names_and_name_ranges() -> None:
    names = [f"Q{index}" for index in range(1, 7)]
    spec = mirt_model("A = Q1-Q3\nB = Q4, Q5 - Q6\nSTART = (Q2, Q4, a1, 1.2)", names)

    assert spec.loadings == ((0, 1, 2), (3, 4, 5))
    assert spec.start[0].items == (1, 3)
    assert spec.item_names == tuple(names)
    assert mirt_model("A = 1-6", names).loadings == ((0, 1, 2, 3, 4, 5),)


def test_numbers_are_item_positions_even_for_numeric_item_names() -> None:
    # Regression: a single number matched an item name before its position,
    # so "1, 3" and "1-3" picked different items for names such as "3", "1".
    names = ["3", "1", "2"]

    assert mirt_model("F = 1, 3", names).loadings == ((0, 2),)
    assert mirt_model("F = 1-2", names).loadings == ((0, 1),)
    assert mirt_model("F = 1-3\nFIXED = (2, a1)", names).fixed[0].items == (1,)


def test_parses_covariance_terms() -> None:
    spec = mirt_model("F1 = 1\nF2 = 2\nF3 = 3\nCOV = F3*F1*F2, F1*F1\nCOV = F2*F1")

    assert spec.covariances == (
        ("F1", "F1"),
        ("F1", "F2"),
        ("F1", "F3"),
        ("F2", "F3"),
    )
    expected = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    assert np.array_equal(spec.covariance_pattern(), expected)


def test_parses_parameter_groups() -> None:
    spec = mirt_model(
        "F1 = 1-4\nF2 = 5-8\n"
        "FIXED = (1, a1), (5-6, d)\n"
        "START = (1, 3, a1, 1.5)\n"
        "PRIOR = (1-8, d, norm, 0, 2)\n"
        "CONSTRAIN = (2-4, a1)"
    )

    assert spec.fixed == (
        ParameterEntry((0,), ("a1",)),
        ParameterEntry((4, 5), ("d",)),
    )
    assert spec.start == (ParameterEntry((0, 2), ("a1",), value=1.5),)
    assert spec.priors[0].prior == ("norm", 0.0, 2.0)
    assert spec.constraints == (ParameterEntry((1, 2, 3), ("a1",)),)
    assert [entry.line for entry in spec.fixed] == [3, 3]


def test_syntax_round_trips_and_repr_is_readable() -> None:
    names = [f"Item_{index}" for index in range(1, 11)]
    spec = mirt_model(
        TWO_FACTORS
        + "\nFIXED = (1, a1)\nSTART = (1, 7, a1, a2, 1.25)\n"
        + "PRIOR = (1-10, d, norm, 0, 2)\nCONSTRAIN = (2-3, a1)",
        item_names=names,
    )

    assert mirt_model(spec.to_syntax(), item_names=names) == spec
    assert str(spec) == spec.to_syntax()
    assert repr(spec).splitlines()[:4] == [
        "ModelSpec(",
        "    F1 = 1-5",
        "    F2 = 6-10",
        "    COV = F1*F2",
    ]
    assert "START = (1, 7, a1, a2, 1.25)" in spec.to_syntax()
    assert len({spec, mirt_model(spec.to_syntax(), item_names=names)}) == 1
    with pytest.raises(AttributeError):
        spec.factors = ("G",)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("syntax", "line", "message"),
    [
        ("F1 = 1-5\nF2 6-10", 2, "expected 'NAME = value'"),
        ("F1 = 1-5\nCOV = F1*F3", 2, "unknown factor 'F3'"),
        ("F1 = 1-5\n\nF1 = 6", 3, "defined twice"),
        ("F1 = 1-5\ncov = F1*F1", 2, "upper case; write COV"),
        ("F1 = 1-5\n\nFIXED = (1, a1", 3, "FIXED is incomplete"),
        ("F1 = 0-3", 1, "invalid item range"),
        ("F1 = 5-3", 1, "invalid item range"),
        ("F1 = Q1", 1, "unknown item 'Q1'; pass item_names"),
        ("F1 = 1, 1", 1, "lists an item more than once"),
        ("F1 =", 1, "F1 has no value"),
        ("F1 = 1-5\nSTART = (1, a1)", 2, "START groups look like"),
        ("F1 = 1-5\nSTART = (1, a1, x)", 2, "expected a number"),
        ("F1 = 1-5\nFIXED = 1, a1", 2, "write groups as"),
        ("F1 = 1-5\nFIXED = (1, 2)", 2, "names no parameter"),
        ("F1 = 1-5\nFIXED = ((1, a1))", 2, "nested parentheses"),
        ("F1 = 1-5\nFIXED = (1, a-1)", 2, "invalid parameter name"),
        ("F1 = 1-5\nPRIOR = (1, a1, gamma, 1, 1)", 2, "unknown prior"),
        ("F1 = 1-5\nPRIOR = (1, d, norm, 0, -1)", 2, "must be positive"),
        ("F1 = 1-5\nPRIOR = (1, g, beta, 0, 1)", 2, "must be positive"),
        ("F1 = 1-5\nCOV = F1*F1*F1", 2, "repeats a factor"),
        ("F1 = 1-5\nCOV = F1", 2, "COV terms look like"),
        ("F1.2 = 1\n2F = 1", 2, "invalid factor name"),
    ],
)
def test_syntax_errors_report_the_line(syntax, line, message) -> None:
    with pytest.raises(MirtValidationError, match=message) as error:
        mirt_model(syntax)

    assert str(error.value).startswith(f"line {line}: ")
    assert error.value.context["line"] == line


def test_item_names_bound_numeric_references() -> None:
    with pytest.raises(MirtValidationError, match="line 1: item 7 is beyond"):
        mirt_model("F = 1-7", item_names=[f"Q{index}" for index in range(6)])
    with pytest.raises(MirtValidationError, match="runs backwards"):
        mirt_model("F = Q3-Q1", item_names=["Q1", "Q2", "Q3"])


def test_rejects_unsupported_keywords_and_empty_models() -> None:
    with pytest.raises(NotImplementedError, match="line 2: MEAN is not supported"):
        mirt_model("F1 = 1-5\nMEAN = F1")
    with pytest.raises(MirtValidationError, match="defines no factors"):
        mirt_model("# nothing here\n")
    with pytest.raises(MirtValidationError, match="syntax must be a string"):
        mirt_model(["F1 = 1-5"])  # type: ignore[arg-type]


def test_model_spec_validates_direct_construction() -> None:
    spec = ModelSpec(factors=["F1", "F2"], loadings=[[2, 0, 1], [3]])
    assert spec.loadings == ((0, 1, 2), (3,))
    assert spec.covariance_pattern().sum() == 0

    with pytest.raises(MirtValidationError, match="keyword"):
        ModelSpec(factors=("COV",), loadings=((0,),))
    with pytest.raises(MirtValidationError, match="unique"):
        ModelSpec(factors=("F", "F"), loadings=((0,), (1,)))
    with pytest.raises(MirtValidationError, match="every factor"):
        ModelSpec(factors=("F1", "F2"), loadings=((0,),))
    with pytest.raises(MirtValidationError, match="zero-based"):
        ModelSpec(factors=("F",), loadings=((-1,),))
    with pytest.raises(MirtValidationError, match="unknown factor"):
        ModelSpec(factors=("F",), loadings=((0,),), covariances=(("F", "G"),))
    with pytest.raises(MirtValidationError, match="START groups need a value"):
        ModelSpec(factors=("F",), loadings=((0,),), start=(ParameterEntry((0,), "a1"),))
    with pytest.raises(MirtValidationError, match="only 1 item names"):
        ModelSpec(factors=("F",), loadings=((0, 1),), item_names=("Q1",))


# Fitting


def test_confirmatory_fit_recovers_factor_correlation(cfa_data, cfa_fit) -> None:
    _, loadings, intercepts = cfa_data
    result = cfa_fit

    assert result.converged
    assert isinstance(result.model, mirt.MultidimensionalModel)
    assert result.latent_covariance[0, 1] == pytest.approx(0.5, abs=0.08)
    assert_allclose(np.diag(result.latent_covariance), 1.0)
    assert_allclose(result.factor_correlation, result.latent_covariance)
    slopes = result.model.parameters["slopes"]
    assert np.all(slopes[loadings == 0] == 0.0)
    assert_allclose(slopes, loadings, atol=0.3)
    assert_allclose(result.model.parameters["intercepts"], intercepts, atol=0.3)
    # Ten slopes, ten intercepts and one correlation.
    assert result.n_parameters == 21
    assert "Latent covariance" in result.summary()


def test_unidimensional_spec_matches_exploratory_fit() -> None:
    data = mirt.simdata(model="2PL", n_persons=800, n_items=6, seed=4)

    exploratory = fit_mirt(data, "2PL")
    confirmatory = fit_mirt(data, "2PL", spec="F = 1-6")

    assert confirmatory.latent_covariance is None
    assert confirmatory.log_likelihood == pytest.approx(
        exploratory.log_likelihood, abs=1e-3
    )
    assert_allclose(
        confirmatory.model.parameters["discrimination"],
        exploratory.model.parameters["discrimination"],
        atol=5e-3,
    )


def test_orthogonal_factors_keep_the_identity_covariance(small_data) -> None:
    result = fit_mirt(
        small_data,
        "2PL",
        spec="F1 = 1-5\nF2 = 6-10",
        n_quadpts=9,
        compute_standard_errors=False,
    )

    assert result.latent_covariance is None
    assert result.n_parameters == 20


def test_fixed_start_and_prior_route_through_fit_keywords() -> None:
    data = mirt.simdata(model="2PL", n_persons=600, n_items=6, seed=8)
    spec = (
        "F = 1-6\n"
        "FIXED = (1, a1)\n"
        "START = (1, a1, 1.3), (2-3, difficulty, 0.5)\n"
        "PRIOR = (1-6, a1, lnorm, 0, 0.5)"
    )
    anchor = np.zeros(6, dtype=bool)
    anchor[0] = True
    discrimination = np.ones(6)
    discrimination[0] = 1.3
    difficulty = np.zeros(6)
    difficulty[1:3] = 0.5

    from_spec = fit_mirt(data, "2PL", spec=spec)
    from_keywords = fit_mirt(
        data,
        "2PL",
        start_values={"discrimination": discrimination, "difficulty": difficulty},
        fixed={"discrimination": anchor},
        priors={"discrimination": LogNormalPrior(0.0, 0.5)},
    )

    assert from_spec.model.parameters["discrimination"][0] == 1.3
    assert from_spec.log_posterior is not None
    assert from_spec.log_posterior == pytest.approx(from_keywords.log_posterior)
    for name, values in from_keywords.model.parameters.items():
        assert_allclose(from_spec.model.parameters[name], values)
    assert from_spec.n_parameters == from_keywords.n_parameters == 11


def test_spec_combines_with_fit_keywords(small_data) -> None:
    intercepts = np.full(10, 0.2)
    fixed = np.zeros(10, dtype=bool)
    fixed[9] = True

    result = fit_mirt(
        small_data,
        "2PL",
        spec=TWO_FACTORS + "\nFIXED = (1, a1)\nSTART = (1, a1, 1.4), (10, d, -0.5)",
        start_values={"intercepts": intercepts},
        fixed={"intercepts": fixed},
        n_quadpts=9,
        compute_standard_errors=False,
    )

    parameters = result.model.parameters
    assert parameters["slopes"][0, 0] == 1.4
    assert parameters["intercepts"][9] == -0.5
    masks = result.model.free_parameter_masks
    assert not masks["slopes"][0, 0] and not masks["intercepts"][9]
    # Variance of F1 stays one: its scale is pinned by the fixed slope.
    assert_allclose(np.diag(result.latent_covariance), 1.0)
    assert result.n_parameters == 18 + 1


def test_free_variance_requires_and_uses_a_fixed_slope() -> None:
    rng = np.random.default_rng(11)
    theta = rng.normal(0, 1.3, 2000)
    difficulty = rng.normal(0, 1, 8)
    logits = 1.5 * (theta[:, None] - difficulty)
    data = (rng.random(logits.shape) < 1 / (1 + np.exp(-logits))).astype(int)

    with pytest.raises(MirtValidationError, match="variance of F"):
        fit_mirt(data, "2PL", spec="F = 1-8\nCOV = F*F")

    anchored = fit_mirt(
        data,
        "2PL",
        spec="F = 1-8\nCOV = F*F\nFIXED = (1, a1)\nSTART = (1, a1, 1.5)",
        n_quadpts=15,
        compute_standard_errors=False,
    )
    rasch = fit_mirt(
        data,
        "1PL",
        spec="F = 1-8\nCOV = F*F",
        n_quadpts=15,
        compute_standard_errors=False,
    )

    assert anchored.latent_covariance[0, 0] == pytest.approx(1.69, abs=0.4)
    assert rasch.latent_covariance[0, 0] == pytest.approx(1.5**2 * 1.69, rel=0.15)
    assert rasch.n_parameters == 8 + 1


@pytest.mark.parametrize("family", ["GRM", "GPCM"])
def test_polytomous_confirmatory_fit_masks_slopes(family) -> None:
    rng = np.random.default_rng(5)
    loadings = np.zeros((6, 2))
    loadings[:3, 0] = rng.uniform(1.2, 2, 3)
    loadings[3:, 1] = rng.uniform(1.2, 2, 3)
    theta = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], 500)
    thresholds = np.sort(rng.normal(0, 1, (6, 2)), axis=1)
    eta = theta @ loadings.T
    cumulative = 1 / (1 + np.exp(-(eta[:, :, None] - thresholds[None])))
    data = (rng.random((500, 6))[:, :, None] < cumulative).sum(axis=2)

    result = fit_mirt(
        data,
        family,
        spec="F1 = 1-3\nF2 = 4-6\nCOV = F1*F2",
        n_quadpts=9,
        compute_standard_errors=False,
    )

    discrimination = result.model.parameters["discrimination"]
    assert discrimination.shape == (6, 2)
    assert np.all(discrimination[loadings == 0] == 0.0)
    assert np.all(discrimination[loadings > 0] > 0.1)
    assert not np.any(
        result.model.free_parameter_masks["discrimination"][loadings == 0]
    )
    assert 0.1 < result.latent_covariance[0, 1] < 0.7


def test_fscores_default_to_the_estimated_covariance(cfa_data, cfa_fit) -> None:
    responses = cfa_data[0][:200]

    default = mirt.fscores(cfa_fit, responses, n_quadpts=15)
    explicit = mirt.fscores(
        cfa_fit, responses, n_quadpts=15, prior_cov=cfa_fit.latent_covariance
    )
    identity = mirt.fscores(cfa_fit, responses, n_quadpts=15, prior_cov=np.eye(2))

    assert_allclose(default.theta, explicit.theta)
    assert not np.allclose(default.theta, identity.theta)
    posterior = mirt.ability_posterior(cfa_fit, responses, n_quadpts=15)
    assert_allclose(posterior.mean, default.theta)


def test_syntax_string_uses_item_names(small_data) -> None:
    names = [f"Q{index}" for index in range(1, 11)]

    result = fit_mirt(
        small_data,
        "2PL",
        item_names=names,
        spec="A = Q1-Q5\nB = Q6-Q10\nCOV = A*B",
        n_quadpts=9,
        compute_standard_errors=False,
    )

    assert result.model.item_names == names
    assert result.latent_covariance.shape == (2, 2)
    with pytest.raises(MirtValidationError, match="differ from the data"):
        fit_mirt(
            small_data,
            "2PL",
            spec=mirt_model("A = 1-10", item_names=[f"I{i}" for i in range(10)]),
            item_names=names,
        )


def test_item_names_must_match_the_data(small_data) -> None:
    # Regression: names shorter than the data raised IndexError.
    short = mirt_model("F1 = 1-5\nF2 = 6-9", item_names=[f"Q{i}" for i in range(9)])

    with pytest.raises(MirtValidationError, match="9 item names, but the data"):
        fit_mirt(small_data, "2PL", spec=short)
    with pytest.raises(MirtValidationError, match="item_names has 9 names"):
        fit_mirt(small_data, "2PL", spec=TWO_FACTORS, item_names=list(short.item_names))


def test_prior_specification_must_apply_to_the_fitted_model(small_data) -> None:
    # Regression: the slope-intercept model fitted for a multidimensional 2PL
    # has no discrimination or difficulty, so these priors were ignored.
    priors = PriorSpecification(discrimination=LogNormalPrior(0.0, 0.5))

    with pytest.raises(MirtValidationError, match="sets no prior on the param"):
        fit_mirt(small_data, "2PL", spec=TWO_FACTORS, priors=priors)
    unidimensional = fit_mirt(
        small_data,
        "2PL",
        spec="F = 1-10",
        priors=priors,
        n_quadpts=9,
        compute_standard_errors=False,
    )
    assert unidimensional.log_posterior is not None


def test_spec_priors_require_full_coverage_and_one_source(small_data) -> None:
    with pytest.raises(MirtValidationError, match="line 4: PRIOR must cover"):
        fit_mirt(
            small_data, "2PL", spec=TWO_FACTORS + "\nPRIOR = (1-4, a1, lnorm, 0, 1)"
        )
    with pytest.raises(MirtValidationError, match="line 5: PRIOR gives 'slopes'"):
        fit_mirt(
            small_data,
            "2PL",
            spec=TWO_FACTORS
            + "\nPRIOR = (1-5, a1, lnorm, 0, 1)\nPRIOR = (6-10, a2, lnorm, 0, 2)",
        )
    with pytest.raises(MirtValidationError, match="not both"):
        fit_mirt(
            small_data,
            "2PL",
            spec=TWO_FACTORS + "\nPRIOR = (1-10, d, norm, 0, 2)",
            priors={"slopes": NormalPrior(0, 1)},
        )


@pytest.mark.parametrize(
    ("syntax", "error", "message"),
    [
        (TWO_FACTORS + "\nCONSTRAIN = (1, 6, a1, a2)", NotImplementedError, "line 4"),
        (TWO_FACTORS + "\nCONSTRAIN = (5-6, a1)", MirtValidationError, "load on"),
        (TWO_FACTORS + "\nCONSTRAIN = (1-5, a3)", MirtValidationError, "factor 3"),
        (TWO_FACTORS + "\nCONSTRAIN = (5-6, a)", MirtValidationError, "only one"),
        (
            TWO_FACTORS + "\nCONSTRAIN = (1-3, a1), (3-5, a1)",
            MirtValidationError,
            "line 4: a1 of Item_3 is already tied",
        ),
        (
            TWO_FACTORS + "\nFIXED = (2, a1)\nCONSTRAIN = (1-3, a1)",
            MirtValidationError,
            "Item_2 .*is fixed",
        ),
        (TWO_FACTORS + "\nSTART = (1, a2, 1.0)", MirtValidationError, "load on"),
        (TWO_FACTORS + "\nFIXED = (1, a3)", MirtValidationError, "factor 3"),
        ("F = 1-10\nFIXED = (1, d)", MirtValidationError, "difficulty"),
        ("F1 = 1-5\nF2 = 6-9", MirtValidationError, "Item_10 load on none"),
        ("F1 = 1-5\nF2 = 6-11", MirtValidationError, "refers to item 11"),
    ],
)
def test_fit_rejects_invalid_specifications(small_data, syntax, error, message):
    with pytest.raises(error, match=message):
        fit_mirt(small_data, "2PL", spec=syntax)


def test_constrain_fits_equal_slopes_through_constraints() -> None:
    data = mirt.simdata(model="2PL", n_persons=600, n_items=6, seed=8)

    from_spec = fit_mirt(data, "2PL", spec="F = 1-6\nCONSTRAIN = (1-4, a1)")
    from_keywords = fit_mirt(
        data, "2PL", constraints=[("discrimination", [0, 1, 2, 3])]
    )

    slopes = from_spec.model.parameters["discrimination"]
    assert np.unique(slopes[:4]).size == 1
    assert from_spec.n_parameters == from_keywords.n_parameters == 12 - 3
    for name, values in from_keywords.model.parameters.items():
        assert_allclose(from_spec.model.parameters[name], values)
    assert_allclose(
        from_spec.standard_errors["discrimination"],
        from_keywords.standard_errors["discrimination"],
    )


def test_constrain_ties_factor_slopes_and_whole_rows(small_data) -> None:
    result = fit_mirt(
        small_data,
        "2PL",
        spec=TWO_FACTORS + "\nCONSTRAIN = (1-5, a1), (6-7, d)",
        n_quadpts=9,
        compute_standard_errors=False,
    )
    slopes = result.model.parameters["slopes"]
    assert np.unique(slopes[:5, 0]).size == 1
    assert np.all(slopes[:5, 1] == 0.0)
    assert np.unique(result.model.parameters["intercepts"][5:7]).size == 1
    assert result.n_parameters == 20 - 4 - 1 + 1

    graded = mirt.simdata(model="GRM", n_persons=400, n_items=4, seed=3)
    both = fit_mirt(
        graded,
        "GRM",
        spec="F = 1-4\nCONSTRAIN = (1-2, thresholds)",
        constraints=[("discrimination", [2, 3])],
        compute_standard_errors=False,
        max_iter=20,
    )
    thresholds = both.model.parameters["thresholds"]
    assert_allclose(thresholds[0], thresholds[1])
    assert both.model.parameters["discrimination"][2] == pytest.approx(
        both.model.parameters["discrimination"][3]
    )


def test_constrain_needs_free_coordinates(small_data) -> None:
    with pytest.raises(MirtValidationError, match="line 2: CONSTRAIN selects no"):
        fit_mirt(small_data[:, :4], "1PL", spec="F = 1-4\nCONSTRAIN = (1-4, a1)")
    with pytest.raises(MirtValidationError, match="at least two items"):
        fit_mirt(small_data, "2PL", spec=TWO_FACTORS + "\nCONSTRAIN = (1, a1)")


def test_fit_rejects_unsupported_settings(small_data) -> None:
    with pytest.raises(MirtValidationError, match="estimation='EM'"):
        fit_mirt(small_data, "2PL", spec=TWO_FACTORS, estimation="MHRM")
    with pytest.raises(MirtValidationError, match="n_factors=3"):
        fit_mirt(small_data, "2PL", spec=TWO_FACTORS, n_factors=3)
    with pytest.raises(MirtModelError, match="supports 2PL, GRM, GPCM"):
        fit_mirt(small_data, "3PL", spec=TWO_FACTORS)
    with pytest.raises(MirtValidationError, match="ModelSpec"):
        fit_mirt(small_data, "2PL", spec=3)  # type: ignore[arg-type]
    with pytest.raises(MirtValidationError, match="must be zero"):
        fit_mirt(
            small_data,
            "GRM",
            spec=TWO_FACTORS,
            start_values={"discrimination": np.ones((10, 2))},
        )


# Results


def test_fit_result_validates_and_serializes_latent_covariance() -> None:
    model = mirt.GradedResponseModel(4, n_categories=3, n_factors=2)
    model._is_fitted = True
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    result = FitResult(
        model=model,
        log_likelihood=-10.0,
        n_iterations=3,
        converged=True,
        standard_errors={},
        aic=30.0,
        bic=40.0,
        latent_covariance=cov.tolist(),
    )

    assert_allclose(result.latent_covariance, cov)
    assert result.factor_correlation[0, 1] == pytest.approx(0.3 / np.sqrt(2.0))
    restored = FitResult.from_json(result.to_json())
    assert_allclose(restored.latent_covariance, cov)
    assert "latent_covariance" not in result.to_dict(include_parameters=False)

    for bad, message in (
        (np.eye(3), "shape"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive definite"),
        (np.array([[1.0, 0.1], [0.2, 1.0]]), "symmetric"),
    ):
        with pytest.raises(MirtValidationError, match=message):
            FitResult(
                model=model,
                log_likelihood=-10.0,
                n_iterations=3,
                converged=True,
                standard_errors={},
                aic=30.0,
                bic=40.0,
                latent_covariance=bad,
            )


def test_public_exports() -> None:
    from mirt import model_syntax

    assert mirt.mirt_model is model_syntax.mirt_model
    assert mirt.ModelSpec is model_syntax.ModelSpec
    assert {"mirt_model", "ModelSpec"} <= set(mirt.__all__)
    assert mirt.estimation.FactorCovarianceDensity is FactorCovarianceDensity


def test_spec_fits_forward_squarem_acceleration():
    import mirt

    data = mirt.simdata("2PL", n_persons=400, n_items=6, seed=11)
    plain = mirt.fit_mirt(data, spec="F1 = 1-6", compute_standard_errors=False)
    fast = mirt.fit_mirt(
        data,
        spec="F1 = 1-6",
        accelerate="squarem",
        compute_standard_errors=False,
    )
    assert fast.n_iterations <= plain.n_iterations
    assert abs(fast.log_likelihood - plain.log_likelihood) < 1e-2
