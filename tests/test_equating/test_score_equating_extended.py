"""Regression tests for model-aware score equating."""

from itertools import product

import numpy as np
import pytest

from mirt.equating import (
    LinkingConstants,
    LinkingResult,
    compute_see,
    equipercentile_equating,
    lord_wingersky_recursion,
    observed_score_equating,
    score_to_theta,
    theta_to_score,
    true_score_equating,
)
from mirt.models.dichotomous import (
    FourParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import GradedResponseModel


@pytest.fixture
def three_pl() -> ThreeParameterLogistic:
    model = ThreeParameterLogistic(4)
    model.set_parameters(
        discrimination=np.array([0.8, 1.0, 1.3, 1.6]),
        difficulty=np.array([-1.0, -0.2, 0.5, 1.2]),
        guessing=np.array([0.15, 0.20, 0.25, 0.30]),
    )
    return model


@pytest.fixture
def four_pl() -> FourParameterLogistic:
    model = FourParameterLogistic(3)
    model.set_parameters(
        discrimination=np.array([0.9, 1.2, 1.5]),
        difficulty=np.array([-0.8, 0.1, 1.0]),
        guessing=np.array([0.10, 0.20, 0.25]),
        upper=np.array([0.90, 0.85, 0.80]),
    )
    return model


@pytest.fixture
def graded_model() -> GradedResponseModel:
    model = GradedResponseModel(3, n_categories=[3, 4, 3])
    model.set_parameters(
        discrimination=np.array([0.8, 1.1, 1.4]),
        thresholds=np.array(
            [
                [-1.2, 0.7, 0.0],
                [-1.5, 0.0, 1.3],
                [-0.8, 1.0, 0.0],
            ]
        ),
    )
    return model


def _direct_polytomous_distribution(
    model: GradedResponseModel,
    theta_grid: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    probabilities = model.probability(theta_grid[:, None])
    category_counts = model.n_categories
    maximum = sum(count - 1 for count in category_counts)
    result = np.zeros(maximum + 1)
    for responses in product(*(range(count) for count in category_counts)):
        conditional = np.ones(len(theta_grid))
        for item_idx, response in enumerate(responses):
            conditional *= probabilities[:, item_idx, response]
        result[sum(responses)] += weights @ conditional
    return result / result.sum()


@pytest.mark.parametrize("model_fixture", ["three_pl", "four_pl", "graded_model"])
def test_theta_to_score_uses_model_probabilities(
    request: pytest.FixtureRequest, model_fixture: str
) -> None:
    model = request.getfixturevalue(model_fixture)
    theta = np.linspace(-3.0, 3.0, 31)

    actual = theta_to_score(model, theta)
    expected = model.expected_score(theta[:, None])

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_theta_to_score_supports_scalar_and_item_subset(graded_model) -> None:
    actual = theta_to_score(graded_model, np.array(0.25), items=[0, 2])
    expected = sum(
        graded_model.expected_score(np.array([[0.25]]), item_idx) for item_idx in (0, 2)
    )

    assert actual.shape == ()
    assert float(actual) == pytest.approx(float(expected[0]))


def test_true_score_equating_supports_polytomous_forms(graded_model) -> None:
    result = true_score_equating(
        graded_model,
        graded_model,
        theta_range=(-10.0, 10.0),
        n_theta=2001,
    )

    np.testing.assert_array_equal(result.old_scores, np.arange(8, dtype=float))
    np.testing.assert_allclose(result.new_scores, result.old_scores, atol=1e-10)


def test_true_score_equating_includes_unreached_endpoint_scores() -> None:
    model = TwoParameterLogistic(5)
    model.set_parameters(
        discrimination=np.ones(5),
        difficulty=np.full(5, 8.0),
    )

    result = true_score_equating(model, model, theta_range=(-4.0, 4.0))

    np.testing.assert_array_equal(result.old_scores, np.arange(6, dtype=float))
    # Abilities for high scores lie beyond theta_range but are still solved.
    np.testing.assert_allclose(result.new_scores, result.old_scores, atol=1e-10)


def test_true_score_equating_rejects_invalid_linking_constants(three_pl) -> None:
    linking = LinkingResult(
        constants=LinkingConstants(A=0.0, B=0.0),
        anchor_items=[0, 1],
    )

    with pytest.raises(ValueError, match="A must be finite and positive"):
        true_score_equating(three_pl, three_pl, linking_result=linking)


@pytest.mark.parametrize("model_fixture", ["three_pl", "four_pl"])
def test_lord_wingersky_uses_asymptotes(
    request: pytest.FixtureRequest, model_fixture: str
) -> None:
    model = request.getfixturevalue(model_fixture)
    theta_grid = np.array([-3.0, -0.5, 1.5])
    weights = np.array([0.2, 0.5, 0.3])

    actual = lord_wingersky_recursion(model, theta_grid, weights, items=[0])
    expected_correct = weights @ model.probability(theta_grid[:, None], 0)

    np.testing.assert_allclose(
        actual, np.array([1.0 - expected_correct, expected_correct]), atol=1e-12
    )


def test_lord_wingersky_matches_direct_polytomous_enumeration(graded_model) -> None:
    theta_grid = np.array([-2.0, -0.25, 1.5])
    weights = np.array([0.25, 0.5, 0.25])

    actual = lord_wingersky_recursion(graded_model, theta_grid, weights)
    expected = _direct_polytomous_distribution(graded_model, theta_grid, weights)

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert len(actual) == 8
    assert actual.sum() == pytest.approx(1.0)


def test_lord_wingersky_normalizes_weights(three_pl) -> None:
    theta_grid = np.linspace(-2.0, 2.0, 9)
    weights = np.arange(1.0, 10.0)

    scaled = lord_wingersky_recursion(three_pl, theta_grid, weights)
    normalized = lord_wingersky_recursion(three_pl, theta_grid, weights / weights.sum())

    np.testing.assert_allclose(scaled, normalized, atol=1e-14)


def test_observed_score_equating_supports_polytomous_forms(graded_model) -> None:
    result = observed_score_equating(graded_model, graded_model)

    np.testing.assert_array_equal(result.old_scores, np.arange(8, dtype=float))
    np.testing.assert_allclose(result.new_scores, result.old_scores, atol=1e-12)


def test_compute_see_uses_three_pl_probabilities(three_pl) -> None:
    theta = np.linspace(-2.0, 2.0, 11)
    probabilities = three_pl.probability(theta[:, None])
    expected = np.sqrt(2.0 * np.sum(probabilities * (1.0 - probabilities), axis=1))

    actual = compute_see(three_pl, three_pl, theta)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_compute_see_uses_polytomous_category_variance(graded_model) -> None:
    theta = np.linspace(-2.0, 2.0, 11)
    probabilities = graded_model.probability(theta[:, None])
    variance = np.zeros(len(theta))
    for item_idx, count in enumerate(graded_model.n_categories):
        scores = np.arange(count, dtype=float)
        item_probabilities = probabilities[:, item_idx, :count]
        mean = item_probabilities @ scores
        variance += item_probabilities @ (scores**2) - mean**2

    actual = compute_see(graded_model, graded_model, theta)

    np.testing.assert_allclose(actual, np.sqrt(2.0 * variance), atol=1e-12)


@pytest.mark.parametrize(
    ("distribution", "message"),
    [
        (np.array([0.4, -0.1, 0.7]), "non-negative"),
        (np.zeros(3), "positive finite mass"),
        (np.array([0.2, np.nan, 0.8]), "finite"),
        (np.ones((2, 2)), "one-dimensional"),
    ],
)
def test_equipercentile_rejects_invalid_distributions(
    distribution: np.ndarray, message: str
) -> None:
    valid = np.array([0.2, 0.3, 0.5])

    with pytest.raises(ValueError, match=message):
        equipercentile_equating(distribution, valid)
    with pytest.raises(ValueError, match=message):
        equipercentile_equating(valid, distribution)


def test_equipercentile_rejects_unknown_smoothing() -> None:
    distribution = np.array([0.2, 0.3, 0.5])

    with pytest.raises(ValueError, match="smoothing must be one of"):
        equipercentile_equating(distribution, distribution, smoothing="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize("smoothing", ["none", "loglinear", "kernel"])
def test_equipercentile_normalizes_without_mutating_inputs(smoothing: str) -> None:
    distribution = np.array([2.0, 3.0, 5.0])
    original = distribution.copy()

    actual = equipercentile_equating(
        distribution,
        distribution,
        smoothing=smoothing,  # type: ignore[arg-type]
    )

    np.testing.assert_allclose(actual, np.arange(3), atol=1e-12)
    np.testing.assert_array_equal(distribution, original)


@pytest.mark.parametrize(
    ("theta_grid", "weights", "message"),
    [
        (np.array([[-1.0, 0.0, 1.0]]), np.ones(3), "one-dimensional"),
        (np.array([-1.0, 0.0, 1.0]), np.ones(2), "same length"),
        (np.array([-1.0, 0.0, 1.0]), np.array([0.5, -0.1, 0.6]), "non-negative"),
        (np.array([-1.0, 0.0, 1.0]), np.zeros(3), "positive finite mass"),
    ],
)
def test_lord_wingersky_rejects_invalid_grid_or_weights(
    three_pl, theta_grid: np.ndarray, weights: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        lord_wingersky_recursion(three_pl, theta_grid, weights)


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([], "non-empty"),
        ([0, 0], "duplicate"),
        ([-1], "outside"),
        ([4], "outside"),
        ([0.0], "integer"),
    ],
)
def test_item_subsets_are_validated(three_pl, items, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        theta_to_score(three_pl, np.array([0.0]), items=items)


def test_score_conversion_rejects_nonmonotonic_model() -> None:
    model = TwoParameterLogistic(2)
    model.set_parameters(
        discrimination=np.array([-1.0, -0.5]),
        difficulty=np.zeros(2),
    )

    with pytest.raises(ValueError, match="non-decreasing"):
        score_to_theta(model, np.array([1.0]))


def test_score_conversion_rejects_nonfinite_values(three_pl) -> None:
    with pytest.raises(ValueError, match="scores must contain only finite"):
        score_to_theta(three_pl, np.array([np.nan]))
    with pytest.raises(ValueError, match="theta must contain only finite"):
        theta_to_score(three_pl, np.array([np.inf]))


@pytest.mark.parametrize(
    ("theta_range", "n_theta", "message"),
    [
        ((1.0, -1.0), 21, "lower bound"),
        ((-1.0, np.inf), 21, "two finite"),
        ((-1.0, 1.0), 1, "at least 2"),
        ((-1.0, 1.0), 2.5, "integer"),
    ],
)
def test_true_score_configuration_is_validated(
    three_pl, theta_range, n_theta, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        true_score_equating(
            three_pl, three_pl, theta_range=theta_range, n_theta=n_theta
        )


def test_multidimensional_models_are_rejected() -> None:
    model = TwoParameterLogistic(3, n_factors=2)

    with pytest.raises(ValueError, match="unidimensional"):
        theta_to_score(model, np.array([0.0]))
    with pytest.raises(ValueError, match="unidimensional"):
        observed_score_equating(model, model)


@pytest.mark.parametrize(
    ("function", "inverse"),
    [
        (lambda x: x**3, np.cbrt),
        (lambda x: np.expm1(8.0 * x), lambda y: np.log1p(y) / 8.0),
        (lambda x: np.tanh(40.0 * x), lambda y: np.arctanh(y) / 40.0),
    ],
)
def test_bracketed_root_solves_steep_and_flat_monotone_curves(function, inverse):
    from mirt.equating.score_equating import _bracketed_root

    targets = np.array([-0.9, -1e-6, 0.0, 0.3, 0.999])
    lower = np.full(len(targets), -1.0)
    upper = np.full(len(targets), 1.0)

    roots = _bracketed_root(
        function, targets, lower, upper, function(lower), function(upper)
    )

    np.testing.assert_allclose(function(roots), targets, atol=1e-13)
    np.testing.assert_allclose(roots, inverse(targets), rtol=1e-9, atol=1e-12)


def _random_graded_form(n_items: int, seed: int) -> GradedResponseModel:
    rng = np.random.default_rng(seed)
    model = GradedResponseModel(n_items, n_categories=5)
    model.set_parameters(
        discrimination=rng.uniform(0.6, 2.0, n_items),
        thresholds=np.sort(rng.normal(0.0, 1.0, (n_items, 4)), axis=1),
    )
    return model


def _random_three_pl_form(n_items: int, seed: int) -> ThreeParameterLogistic:
    rng = np.random.default_rng(seed)
    model = ThreeParameterLogistic(n_items)
    model.set_parameters(
        discrimination=rng.uniform(0.6, 2.0, n_items),
        difficulty=rng.normal(0.0, 1.0, n_items),
        guessing=rng.uniform(0.1, 0.25, n_items),
    )
    return model


def _true_score_oracle(old, new, linking, score):
    """Solve the old curve with Brent's method and read the new curve there."""
    from scipy.optimize import brentq

    def expected(model, theta):
        probabilities = model.probability(np.array([[theta]]))[0]
        if probabilities.ndim == 1:
            return float(np.sum(probabilities))
        return float(np.sum(probabilities * np.arange(probabilities.shape[1])))

    root = brentq(
        lambda theta: expected(old, theta) - score,
        -60.0,
        60.0,
        xtol=1e-15,
        rtol=1e-15,
        maxiter=500,
    )
    A, B = linking.constants.A, linking.constants.B
    return expected(new, (root - B) / A)


@pytest.mark.parametrize("family", ["grm", "3pl"])
def test_true_score_equating_matches_brent_oracle(family: str) -> None:
    """Seeded root finding keeps equated scores exact to rounding level."""
    make_form = _random_graded_form if family == "grm" else _random_three_pl_form
    old, new = make_form(12, 1), make_form(12, 2)
    linking = LinkingResult(LinkingConstants(A=1.2, B=-0.3), anchor_items=[])

    result = true_score_equating(old, new, linking_result=linking)

    lower = float(np.sum(getattr(old, "guessing", np.zeros(old.n_items))))
    interior = (result.old_scores > lower) & (result.old_scores < result.old_scores[-1])
    expected = [
        _true_score_oracle(old, new, linking, score)
        for score in result.old_scores[interior]
    ]
    np.testing.assert_allclose(result.new_scores[interior], expected, atol=1e-10)


def test_true_score_roots_need_about_one_curve_evaluation() -> None:
    """Grid-seeded roots usually evaluate the old form once per score.

    The previous unseeded false-position search needed close to six.
    """
    old, new = _random_graded_form(30, 1), _random_graded_form(30, 2)
    sizes: list[int] = []
    probability = old.probability

    def counted(theta, item_idx=None):
        sizes.append(len(theta))
        return probability(theta, item_idx)

    old.probability = counted  # type: ignore[method-assign]
    result = true_score_equating(old, new)

    n_interior = len(result.old_scores) - 2
    # The first call validates the curve on the search grid.
    assert sum(sizes[1:]) <= 1.5 * n_interior


def test_lagrange_derivatives_are_exact_at_and_between_nodes() -> None:
    from mirt.equating.score_equating import _lagrange_weights

    nodes = np.tile(np.array([0.0, 1.0, 2.0, 3.0]), (3, 1))
    points = np.array([1.0, 1.5, 3.0])

    values = _lagrange_weights(nodes, points) @ (nodes[0] ** 3)
    slopes = _lagrange_weights(nodes, points, derivative=True) @ (nodes[0] ** 3)

    np.testing.assert_allclose(values, points**3, atol=1e-12)
    np.testing.assert_allclose(slopes, 3.0 * points**2, atol=1e-12)


def test_seeded_root_solver_converges_from_inexact_inverse() -> None:
    from mirt.equating.score_equating import _solve_bracketed

    targets = np.array([-0.9, -1e-6, 0.0, 0.3, 0.999])
    lower = np.full(len(targets), -1.0)
    upper = np.full(len(targets), 1.0)

    def function(x):
        return np.expm1(8.0 * x)

    def inverse(values, rows):
        # A deliberately biased approximate inverse.
        return np.log1p(values) / 8.0 * (1.0 + 1e-3) + 1e-4

    roots, residuals = _solve_bracketed(
        function,
        targets,
        lower,
        upper,
        function(lower),
        function(upper),
        inverse=inverse,
    )

    np.testing.assert_allclose(function(roots), targets, atol=1e-13)
    np.testing.assert_allclose(residuals, function(roots) - targets, atol=1e-15)
    np.testing.assert_allclose(roots, np.log1p(targets) / 8.0, rtol=1e-9, atol=1e-12)


def test_loglinear_smoothing_preserves_four_moments() -> None:
    """The log-linear option is the moment-preserving ML presmoother."""
    from mirt.equating.score_equating import _loglinear_smooth

    rng = np.random.default_rng(4)
    counts = rng.multinomial(400, np.r_[np.full(10, 0.06), np.full(11, 0.0364)])
    distribution = counts / counts.sum()

    smoothed = _loglinear_smooth(distribution)

    scores = np.arange(len(distribution), dtype=np.float64)
    for power in range(1, 5):
        assert smoothed @ scores**power == pytest.approx(
            distribution @ scores**power, rel=1e-8
        )
    assert smoothed.sum() == pytest.approx(1.0)
    assert np.all(smoothed > 0.0)
    # A fifth moment is not constrained.
    assert smoothed @ scores**5 != pytest.approx(distribution @ scores**5, rel=1e-8)


@pytest.mark.parametrize(
    ("distribution", "degree"),
    [([0.2, 0.3, 0.5], 1), ([0.5, 0.0, 0.5, 0.0], 1), ([0.4, 0.6], 0)],
)
def test_loglinear_smoothing_lowers_its_degree_for_short_distributions(
    distribution, degree
) -> None:
    from mirt.equating.score_equating import _loglinear_smooth

    values = np.asarray(distribution, dtype=np.float64)
    smoothed = _loglinear_smooth(values)

    scores = np.arange(len(values), dtype=np.float64)
    if degree == 0:
        np.testing.assert_array_equal(smoothed, values)
    else:
        assert smoothed @ scores == pytest.approx(values @ scores, rel=1e-10)


class _UnusableLastItem2PL(TwoParameterLogistic):
    """A 2PL whose last item, e.g. an uncalibrated bank item, has NaN curves."""

    def probability(self, theta, item_idx=None):
        probabilities = np.array(super().probability(theta, item_idx), dtype=float)
        if item_idx is None:
            probabilities[..., -1] = np.nan
        return probabilities


class _UnusableGraded(GradedResponseModel):
    """A GRM with NaN curves for its last item and its padded categories."""

    def probability(self, theta, item_idx=None):
        probabilities = np.array(super().probability(theta, item_idx), dtype=float)
        if item_idx is None:
            probabilities[:, -1, :] = np.nan
            for item, count in enumerate(self.n_categories):
                probabilities[:, item, count:] = np.nan
        return probabilities


def test_true_score_roots_ignore_unselected_items() -> None:
    """Only selected items' curves enter the root search, as in validation."""
    slopes_old, slopes_new = np.linspace(0.8, 1.6, 6), np.linspace(0.9, 1.5, 6)
    locations_old = np.linspace(-1.0, 1.0, 6)
    locations_new = np.linspace(-1.2, 0.8, 6)
    forms = []
    for model_type in (_UnusableLastItem2PL, TwoParameterLogistic):
        old, new = model_type(6), model_type(6)
        old.set_parameters(discrimination=slopes_old, difficulty=locations_old)
        new.set_parameters(discrimination=slopes_new, difficulty=locations_new)
        forms.append((old, new))
    items = [0, 1, 2, 3, 4]

    result = true_score_equating(*forms[0], items_old=items, items_new=items)
    reference = true_score_equating(*forms[1], items_old=items, items_new=items)

    np.testing.assert_array_equal(result.new_scores, reference.new_scores)
    np.testing.assert_allclose(
        result.new_scores,
        [0.0, 1.141316, 2.272009, 3.292744, 4.199456, 5.0],
        atol=5e-7,
    )


def test_true_score_roots_ignore_unselected_and_padded_categories() -> None:
    counts = [3, 5, 4, 5, 5]
    forms = []
    for model_type in (_UnusableGraded, GradedResponseModel):
        pair = []
        for seed in (1, 2):
            rng = np.random.default_rng(seed)
            model = model_type(5, n_categories=counts)
            thresholds = np.sort(rng.normal(0.0, 1.0, (5, 4)), axis=1)
            model.set_parameters(
                discrimination=rng.uniform(0.6, 2.0, 5), thresholds=thresholds
            )
            pair.append(model)
        forms.append(pair)
    items = [0, 1, 2, 3]

    result = true_score_equating(*forms[0], items_old=items, items_new=items)
    reference = true_score_equating(*forms[1], items_old=items, items_new=items)

    np.testing.assert_allclose(
        result.new_scores, reference.new_scores, rtol=0.0, atol=1e-12
    )


def test_true_score_roots_reject_invalid_probabilities() -> None:
    """Curves are checked at the roots, not only on the validation grid."""
    old, new = _random_three_pl_form(8, 1), _random_three_pl_form(8, 2)
    probability = old.probability
    calls = []

    def inflated_after_validation(theta, item_idx=None):
        calls.append(len(theta))
        values = np.array(probability(theta, item_idx), dtype=float)
        return values if len(calls) == 1 else 1.5 * values

    old.probability = inflated_after_validation  # type: ignore[method-assign]

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        true_score_equating(old, new)
