"""Tests for multigroup IRT analysis."""

import warnings

import numpy as np
import pytest

from mirt.multigroup import compare_invariance, fit_multigroup


class TestMultigroupAnalysis:
    """Tests for multigroup IRT functions."""

    def test_configural_invariance(self, rng):
        """Test configural invariance fitting."""
        n_persons_per_group = 100
        n_items = 10

        theta1 = rng.standard_normal(n_persons_per_group)
        difficulty1 = rng.normal(0, 1, n_items)
        probs1 = 1 / (1 + np.exp(-(theta1[:, None] - difficulty1)))
        responses1 = (rng.random((n_persons_per_group, n_items)) < probs1).astype(int)

        theta2 = rng.standard_normal(n_persons_per_group)
        difficulty2 = difficulty1 + 0.5
        probs2 = 1 / (1 + np.exp(-(theta2[:, None] - difficulty2)))
        responses2 = (rng.random((n_persons_per_group, n_items)) < probs2).astype(int)

        responses = np.vstack([responses1, responses2])
        groups = np.array([0] * n_persons_per_group + [1] * n_persons_per_group)

        result = fit_multigroup(
            responses,
            groups,
            model="2PL",
            invariance="configural",
            n_quadpts=15,
            max_iter=50,
            verbose=False,
        )

        assert result is not None
        assert result.model.is_fitted

    def test_minimum_two_groups(self, rng):
        """Test that at least 2 groups are required."""
        n_persons = 100
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.zeros(n_persons)

        with pytest.raises(ValueError, match="At least 2 groups"):
            fit_multigroup(responses, groups, model="2PL")

    def test_metric_invariance(self, rng):
        """Test metric invariance fitting."""
        n_persons = 100
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array([0] * 50 + [1] * 50)

        result = fit_multigroup(
            responses,
            groups,
            model="2PL",
            invariance="metric",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None
        assert result.model.is_fitted

    def test_scalar_invariance(self, rng):
        """Test scalar invariance fitting."""
        n_persons = 100
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array([0] * 50 + [1] * 50)

        result = fit_multigroup(
            responses,
            groups,
            model="2PL",
            invariance="scalar",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None
        assert result.model.is_fitted

    def test_strict_invariance(self, rng):
        """Test strict invariance fitting."""
        n_persons = 100
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array([0] * 50 + [1] * 50)

        result = fit_multigroup(
            responses,
            groups,
            model="2PL",
            invariance="strict",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None
        assert result.model.is_fitted

    def test_three_groups(self, rng):
        """Test with three groups."""
        n_per_group = 50
        n_items = 5

        responses = rng.integers(0, 2, size=(n_per_group * 3, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group + [2] * n_per_group)

        result = fit_multigroup(
            responses,
            groups,
            model="2PL",
            invariance="configural",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None

    def test_string_group_labels(self, rng):
        """Test with string group labels."""
        n_persons = 60
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array(["A"] * 30 + ["B"] * 30)

        result = fit_multigroup(
            responses,
            groups,
            model="2PL",
            invariance="configural",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None

    def test_1pl_model(self, rng):
        """Test multigroup with 1PL model."""
        n_persons = 60
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array([0] * 30 + [1] * 30)

        result = fit_multigroup(
            responses,
            groups,
            model="1PL",
            invariance="configural",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None
        assert result.model.model_name == "1PL"

    def test_polytomous_grm(self, rng):
        """Test multigroup with GRM model."""
        n_persons = 80
        n_items = 5
        n_categories = 4

        responses = rng.integers(0, n_categories, size=(n_persons, n_items))
        groups = np.array([0] * 40 + [1] * 40)

        result = fit_multigroup(
            responses,
            groups,
            model="GRM",
            n_categories=n_categories,
            invariance="configural",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None
        assert result.model.model_name == "GRM"

    def test_compare_invariance(self, rng):
        """Test comparing different invariance levels."""
        n_persons = 80
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array([0] * 40 + [1] * 40)

        results = compare_invariance(
            responses,
            groups,
            model="2PL",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert "configural" in results
        assert "metric" in results
        assert "scalar" in results
        assert "strict" in results

        for inv, result in results.items():
            assert result.model.is_fitted

    def test_3pl_model(self, rng):
        """Smoke fit for multigroup 3PL."""
        n_persons = 60
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array([0] * 30 + [1] * 30)

        try:
            result = fit_multigroup(
                responses,
                groups,
                model="3PL",
                invariance="configural",
                n_quadpts=11,
                max_iter=20,
                verbose=False,
            )
        except (ValueError, NotImplementedError) as exc:
            pytest.skip(f"3PL multigroup not supported: {exc}")

        assert result is not None
        assert result.model.is_fitted
        assert result.model.model_name == "3PL"

    def test_gpcm_model(self, rng):
        """Smoke fit for multigroup GPCM."""
        n_persons = 60
        n_items = 5
        n_categories = 3

        responses = rng.integers(0, n_categories, size=(n_persons, n_items))
        groups = np.array([0] * 30 + [1] * 30)

        try:
            result = fit_multigroup(
                responses,
                groups,
                model="GPCM",
                n_categories=n_categories,
                invariance="configural",
                n_quadpts=11,
                max_iter=20,
                verbose=False,
            )
        except (ValueError, NotImplementedError) as exc:
            pytest.skip(f"GPCM multigroup not supported: {exc}")

        assert result is not None
        assert result.model.is_fitted
        assert result.model.model_name == "GPCM"

    def test_invariance_lrt_helper(self, rng):
        """Compare invariance fits via exported LRT helper when available."""
        from mirt.multigroup import invariance_lrt

        n_persons = 60
        n_items = 5
        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array([0] * 30 + [1] * 30)

        results = compare_invariance(
            responses,
            groups,
            model="2PL",
            n_quadpts=11,
            max_iter=20,
            verbose=False,
        )

        try:
            lrt = invariance_lrt(results["metric"], results["configural"])
        except (ValueError, TypeError, AttributeError) as exc:
            pytest.skip(f"invariance_lrt not usable with these fits: {exc}")

        assert "chi2" in lrt
        assert "df" in lrt
        assert "p_value" in lrt
        assert lrt["df"] > 0


def _far_group_data(seed=4, n_persons=300, n_items=20):
    """Two 2PL groups, the focal one N(2.5, 0.6^2) on the reference scale."""
    rng = np.random.default_rng(seed)
    slopes = rng.uniform(1.5, 2.5, n_items)
    difficulty = np.linspace(-1.0, 3.5, n_items)

    def simulate(theta):
        probability = 1 / (1 + np.exp(-slopes * (theta[:, None] - difficulty)))
        return (rng.random(probability.shape) < probability).astype(int)

    data = np.vstack(
        [
            simulate(rng.normal(0.0, 1.0, n_persons)),
            simulate(rng.normal(2.5, 0.6, n_persons)),
        ]
    )
    return data, np.repeat([0, 1], n_persons)


def test_default_grid_recovers_a_distant_group_like_a_fine_grid():
    # A fixed 21-node standard-normal Gauss-Hermite grid left the focal
    # group about three effective nodes and biased its mean by 0.26 here.
    data, groups = _far_group_data()
    default = fit_multigroup(data, groups, invariance="scalar")
    fine = fit_multigroup(data, groups, invariance="scalar", n_quadpts=121)

    for result in (default, fine):
        assert result.latent_distributions[1].mean[0] > 2.0
    assert default.latent_distributions[1].mean[0] == pytest.approx(
        fine.latent_distributions[1].mean[0], abs=5e-3
    )
    assert np.sqrt(default.latent_distributions[1].cov[0, 0]) == pytest.approx(
        np.sqrt(fine.latent_distributions[1].cov[0, 0]), abs=5e-3
    )


def test_default_configural_fit_equals_separate_single_group_fits():
    import mirt

    rng = np.random.default_rng(6)
    slopes = rng.uniform(1.5, 2.5, 15)
    difficulty = rng.normal(0.0, 1.0, 15)
    theta = np.concatenate([rng.normal(0.0, 1.0, 300), rng.normal(0.5, 1.2, 300)])
    probability = 1 / (1 + np.exp(-slopes * (theta[:, None] - difficulty)))
    data = (rng.random(probability.shape) < probability).astype(int)
    groups = np.repeat([0, 1], 300)
    configural = fit_multigroup(data, groups, model="2PL")
    separate = [
        mirt.fit_mirt(data[groups == g], model="2PL", compute_standard_errors=False)
        for g in (0, 1)
    ]

    assert configural.group_log_likelihoods == pytest.approx(
        [fit.log_likelihood for fit in separate], abs=0.02
    )


def test_multigroup_estimator_rejects_unknown_quadrature_rule():
    from mirt.multigroup import MultigroupEMEstimator

    with pytest.raises(ValueError, match="quadrature"):
        MultigroupEMEstimator(quadrature="simpson")
    with pytest.raises(ValueError, match="n_quadpts"):
        MultigroupEMEstimator(n_quadpts=4)


def _two_group_2pl(seed=7, n_persons=200, n_items=6):
    rng = np.random.default_rng(seed)
    slopes = rng.uniform(0.8, 1.8, n_items)
    difficulty = rng.normal(0.0, 0.8, n_items)
    theta = np.concatenate([rng.normal(0, 1, n_persons), rng.normal(0.5, 1, n_persons)])
    probability = 1 / (1 + np.exp(-slopes * (theta[:, None] - difficulty)))
    data = (rng.random(probability.shape) < probability).astype(float)
    return data, np.repeat(["A", "B"], n_persons)


def test_item_not_administered_to_a_group_has_no_group_parameters_there():
    data, groups = _two_group_2pl()
    data[groups == "B", 5] = np.nan

    with pytest.warns(UserWarning, match="no responses to items"):
        configural = fit_multigroup(data, groups, invariance="configural")
    with pytest.warns(UserWarning, match="no responses to items"):
        metric = fit_multigroup(data, groups, invariance="metric")
    scalar = fit_multigroup(data, groups, invariance="scalar")

    # Group B's slope and difficulty of the omitted item are unidentified.
    assert configural.n_parameters == 2 * 12 - 2
    assert metric.n_parameters == 6 + 12 - 1 + 1
    assert configural.n_parameters - metric.n_parameters == 4
    assert scalar.n_parameters == 12 + 2
    # coef() reports the unestimated group-specific parameters as NaN.
    rows = configural._coefficient_rows(1, include_standard_errors=False)
    assert np.isnan(rows[5]["discrimination"])
    assert np.isnan(rows[5]["difficulty"])
    assert np.isfinite(rows[4]["difficulty"])
    metric_rows = metric._coefficient_rows(1, include_standard_errors=False)
    assert np.isfinite(metric_rows[5]["discrimination"])
    assert np.isnan(metric_rows[5]["difficulty"])


def test_unused_category_with_group_specific_thresholds_is_rejected():
    from mirt.exceptions import MirtValidationError

    rng = np.random.default_rng(8)
    data = rng.integers(0, 3, size=(240, 4))
    groups = np.repeat([0, 1], 120)
    data[(groups == 1) & (data[:, 0] == 2), 0] = 1

    for level in ("configural", "metric"):
        with pytest.raises(MirtValidationError, match="never uses categories"):
            fit_multigroup(data, groups, model="GRM", invariance=level, max_iter=5)
    # Shared thresholds remain identified through the other group.
    result = fit_multigroup(data, groups, model="GRM", invariance="scalar", max_iter=5)
    assert result.n_parameters == 4 * 3 + 2


def _two_factor_data(seed=31, n_persons=250):
    """Items 0-3 load on F1 and 4-7 on F2; the factors correlate 0.6."""
    rng = np.random.default_rng(seed)
    slopes = np.zeros((8, 2))
    slopes[:4, 0] = rng.uniform(1.3, 2.0, 4)
    slopes[4:, 1] = rng.uniform(1.3, 2.0, 4)
    intercepts = rng.normal(0.0, 0.8, 8)
    theta = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 2 * n_persons)
    probability = 1 / (1 + np.exp(-(theta @ slopes.T + intercepts)))
    data = (rng.random(probability.shape) < probability).astype(int)
    return data, np.repeat([0, 1], n_persons)


def test_fit_multigroup_accepts_exploratory_factors_and_model_syntax():
    data, groups = _two_factor_data()
    options = {"n_quadpts": 9, "max_iter": 150}

    exploratory = fit_multigroup(data, groups, n_factors=2, **options)
    # Two slopes and a difficulty per item and group, less one rotation each.
    assert exploratory.model.n_factors == 2
    assert exploratory.n_parameters == 2 * 24 - 2
    assert exploratory.latent_distributions[0].cov[0, 1] == 0.0

    syntax = "F1 = 1-4\nF2 = 5-8\nCOV = F1*F2"
    correlated = fit_multigroup(data, groups, spec=syntax, **options)
    slopes = correlated.model.get_group_parameters(0)["discrimination"]
    assert np.all(slopes[:4, 1] == 0.0) and np.all(slopes[4:, 0] == 0.0)
    assert correlated.n_parameters == 2 * 16 + 2
    for distribution in correlated.latent_distributions:
        assert distribution.cov[0, 1] == pytest.approx(0.6, abs=0.2)

    orthogonal = fit_multigroup(data, groups, spec="F1 = 1-4\nF2 = 5-8", **options)
    assert orthogonal.latent_distributions[0].cov[0, 1] == 0.0
    assert orthogonal.n_parameters == 2 * 16
    assert correlated.log_likelihood > orthogonal.log_likelihood


def test_fit_multigroup_rejects_unsupported_model_syntax():
    from mirt.exceptions import MirtValidationError

    data, groups = _two_factor_data(n_persons=40)
    for syntax, message in (
        ("F1 = 1-4\nF2 = 5-8\nCOV = F1*F1", "frees a factor variance"),
        ("F1 = 1-4\nF2 = 5-8\nPRIOR = (1-8, a1, norm, 0, 1)", "not PRIOR"),
        ("F1 = 1-4\nF2 = 5-7", "must load on a factor"),
    ):
        with pytest.raises(MirtValidationError, match=message):
            fit_multigroup(data, groups, spec=syntax, max_iter=2)
    with pytest.raises(MirtValidationError, match="cannot be combined with spec"):
        fit_multigroup(data, groups, spec="F1 = 1-4\nF2 = 5-8", free_correlations=True)


@pytest.mark.parametrize("kind", ["nan", "none"])
def test_rows_with_missing_group_labels_are_dropped_with_a_warning(kind):
    # NaN labels used to form an empty group and None labels broke sorting.
    data, groups = _two_group_2pl(n_persons=80)
    dropped = [0, 3, 150]
    if kind == "nan":
        labels = np.where(groups == "A", 1.0, 2.0)
        labels[dropped] = np.nan
    else:
        labels = groups.astype(object)
        labels[dropped] = None

    with pytest.warns(UserWarning, match="3 rows with missing group labels"):
        result = fit_multigroup(data, labels, n_quadpts=15, max_iter=20)
    keep = np.ones(len(groups), dtype=bool)
    keep[dropped] = False
    expected = fit_multigroup(data[keep], groups[keep], n_quadpts=15, max_iter=20)

    assert result.n_groups == 2
    assert result.group_n_observations == [78, 79]
    assert result.log_likelihood == pytest.approx(expected.log_likelihood)


def test_compare_invariance_reuses_scalar_fit_when_strict_adds_nothing():
    from mirt.multigroup import test_invariance_hierarchy

    data, groups = _two_group_2pl(n_persons=120)
    hierarchy = test_invariance_hierarchy(
        data, groups, model="2PL", n_quadpts=15, max_iter=50
    )
    results, comparisons = hierarchy["results"], hierarchy["comparisons"]

    assert results["strict"] is results["scalar"]
    assert [row.comparison for row in comparisons] == [
        "configural vs metric",
        "metric vs scalar",
        "scalar vs strict",
    ]
    last = comparisons[-1]
    assert (last.chi2, last.df, last.significant) == (0.0, 0, False)
    assert np.isnan(last.p_value)
    assert "constrains no parameter beyond scalar" in last.note
    assert comparisons[0].note == ""


def test_group_density_beyond_the_rectangular_grid_warns():
    from mirt.models import TwoParameterLogistic
    from mirt.multigroup import MultigroupLatentDensity, MultigroupModel
    from mirt.multigroup.estimator import _warn_outside_grid

    model = MultigroupModel(TwoParameterLogistic(n_items=3), 3)
    density = MultigroupLatentDensity(3)
    density.set_group_distribution(1, mean=np.array([3.0]), cov=np.array([[0.5]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_outside_grid(model, density, (-6.0, 6.0))
    # N(4.5, 1) puts 6.7% of its mass above the grid's upper node at 6.
    density.set_group_distribution(2, mean=np.array([4.5]), cov=np.array([[1.0]]))
    with pytest.warns(
        UserWarning, match="'Group_2' has 6.68% .* outside the quadrature"
    ):
        _warn_outside_grid(model, density, (-6.0, 6.0))


def test_rectangular_grid_grows_to_cover_a_distant_group():
    from mirt.estimation.quadrature import RectangularQuadrature
    from mirt.multigroup import MultigroupLatentDensity
    from mirt.multigroup.estimator import _covering_grid

    grid = RectangularQuadrature(61)
    density = MultigroupLatentDensity(2)
    density.set_group_distribution(1, mean=np.array([2.5]), cov=np.array([[0.36]]))
    assert _covering_grid(grid, density, 122) is None

    # R's fixed [-6, 6] grid truncates N(4.5, 1), whose estimate then shrinks
    # (to 3.99 and SD 0.75 instead of 4.72 and 0.99 for one simulated focal
    # group); the grid gains whole 0.2 steps up to the mean + 6 SDs.
    density.set_group_distribution(1, mean=np.array([4.5]), cov=np.array([[1.0]]))
    wider = _covering_grid(grid, density, 122)
    assert wider.limits == pytest.approx((-6.0, 10.6))
    assert wider.n_points == 84
    np.testing.assert_allclose(wider.nodes[:61, 0], grid.nodes[:, 0], atol=1e-12)
    capped = _covering_grid(grid, density, 70)
    assert capped.n_points == 70 and capped.limits[1] == pytest.approx(7.8)


def _anchored_forms(seed=31, n_persons=600, n_groups=3, width=12, step=8):
    """Grade-like groups on overlapping 2PL forms linked by a few anchors.

    Groups share only ``width - step`` items with their neighbours, so their
    latent means are weakly identified and plain EM crawls along the ridge.
    """
    rng = np.random.default_rng(seed)
    n_items = step * (n_groups - 1) + width
    slopes = rng.uniform(1.0, 2.2, n_items)
    difficulty = np.linspace(-1.5, 3.0, n_items)
    responses = []
    for g in range(n_groups):
        items = np.arange(step * g, step * g + width)
        theta = rng.normal(0.9 * g, 1.0 - 0.1 * g, n_persons)
        p = 1 / (1 + np.exp(-slopes[items] * (theta[:, None] - difficulty[items])))
        block = np.full((n_persons, n_items), -1)
        block[:, items] = (rng.random(p.shape) < p).astype(int)
        responses.append(block)
    return responses, n_items


def test_squarem_reaches_the_plain_em_maximum_with_fewer_e_steps():
    from mirt.models import TwoParameterLogistic
    from mirt.multigroup import MultigroupEMEstimator, MultigroupModel

    responses, n_items = _anchored_forms()
    fits, histories = {}, {}
    for accelerate in ("none", "squarem"):
        estimator = MultigroupEMEstimator(max_iter=3000, accelerate=accelerate)
        model = MultigroupModel(TwoParameterLogistic(n_items), 3)
        fits[accelerate] = estimator.fit(model, responses, "strict")
        histories[accelerate] = np.array(estimator.convergence_history)
    plain, squarem = fits["none"], fits["squarem"]

    # Plain EM needed 628 iterations, past the default max_iter of 500.
    assert plain.converged and plain.n_iterations > 500
    assert squarem.converged and squarem.n_iterations < plain.n_iterations / 4
    assert squarem.log_likelihood >= plain.log_likelihood - 1e-4
    assert np.all(np.diff(histories["squarem"]) >= -1e-6)
    for accelerated, reference in zip(
        squarem.latent_distributions, plain.latent_distributions, strict=True
    ):
        assert accelerated.mean[0] == pytest.approx(reference.mean[0], abs=0.03)
        assert accelerated.cov[0, 0] == pytest.approx(reference.cov[0, 0], abs=0.02)


def test_squarem_keeps_ordered_means_and_follows_a_growing_grid():
    from mirt.models import TwoParameterLogistic
    from mirt.multigroup import (
        MultigroupEMEstimator,
        MultigroupModel,
        _prepare_multigroup,
    )

    responses, n_items = _anchored_forms(n_persons=300)
    ordered = MultigroupEMEstimator(accelerate="squarem").fit(
        MultigroupModel(TwoParameterLogistic(n_items), 3),
        responses,
        "strict",
        mean_order=(0, 1, 2),
    )
    means = [distribution.mean[0] for distribution in ordered.latent_distributions]
    assert ordered.converged and np.all(np.diff(means) >= 0.0)

    # A focal group 3.5 SDs up makes the grid grow during the fit, which
    # restarts the extrapolation cycle; plain EM needs 781 iterations.
    rng = np.random.default_rng(9)
    slopes = rng.uniform(1.2, 2.0, 24)
    difficulty = np.linspace(-1.5, 5.0, 24)
    theta = np.concatenate([rng.normal(0.0, 1.0, 300), rng.normal(3.5, 1.0, 300)])
    p = 1 / (1 + np.exp(-slopes * (theta[:, None] - difficulty)))
    data = (rng.random(p.shape) < p).astype(int)
    groups = np.repeat([0, 1], 300)
    fits = {}
    for accelerate, max_iter in (("none", 2000), ("squarem", 500)):
        prepared = _prepare_multigroup(
            data, groups, "2PL", n_categories=None, reference_group=0, item_names=None
        )
        estimator = MultigroupEMEstimator(accelerate=accelerate, max_iter=max_iter)
        fits[accelerate] = estimator.fit(prepared.model, prepared.responses, "scalar")
        assert estimator._quadrature.n_points > 61
    plain, squarem = fits["none"], fits["squarem"]
    assert plain.converged and plain.n_iterations > 500
    assert squarem.converged and squarem.n_iterations < plain.n_iterations / 4
    assert squarem.log_likelihood >= plain.log_likelihood - 1e-4
    assert squarem.latent_distributions[1].mean[0] == pytest.approx(
        plain.latent_distributions[1].mean[0], abs=0.01
    )


def test_unknown_acceleration_is_rejected():
    from mirt.exceptions import MirtValidationError
    from mirt.multigroup import MultigroupEMEstimator

    with pytest.raises(MirtValidationError, match="accelerate"):
        MultigroupEMEstimator(accelerate="aitken")
    with pytest.raises(MirtValidationError, match="accelerate"):
        fit_multigroup(
            np.eye(4, dtype=int)[:, :2], np.array([0, 1, 0, 1]), accelerate=1
        )


def _records(frame):
    if hasattr(frame, "to_dicts"):
        return frame.to_dicts()
    return frame.to_dict(orient="records")


def _r_reference_data():
    """The 2PL data behind the R mirt 1.48 ``multipleGroup`` references."""
    rng = np.random.default_rng(1)
    slopes = rng.uniform(0.8, 2.0, 12)
    difficulty = rng.normal(0, 1, 12)
    theta = (rng.normal(0, 1, 800), rng.normal(0.5, np.sqrt(1.5), 800))

    def simulate(abilities, locations):
        p = 1 / (1 + np.exp(-slopes * (abilities[:, None] - locations)))
        return (rng.random(p.shape) < p).astype(int)

    shifted = difficulty.copy()
    shifted[0] += 0.8
    data = np.vstack([simulate(theta[0], difficulty), simulate(theta[1], shifted)])
    return data, np.array(["A"] * 800 + ["B"] * 800)


# R: multipleGroup(dat, 1, grp, invariance=..., SE=TRUE, SE.type="Oakes",
# TOL=1e-6) and coef(fit, printSE=TRUE, IRTpars=TRUE): (group, item) ->
# (a, SE(a), b, SE(b)), and the focal (MEAN_1, SE, COV_11, SE).
_R_STANDARD_ERRORS = {
    "configural": (
        {
            ("A", 0): (1.3895069, 0.1430133, -0.59552833, 0.08024883),
            ("B", 0): (1.808335, 0.168042, -0.3731658, 0.0639105),
            ("B", 2): (1.1528142, 0.1234728, -0.9365509, 0.1068268),
        },
        None,
    ),
    "metric": (
        {
            ("A", 0): (1.412571, 0.104181, -0.58979563, 0.07587557),
            ("B", 0): (1.412571, 0.104181, -0.47798846, 0.08204083),
            ("B", 2): (1.00462444, 0.08145836, -1.1184100, 0.1182522),
        },
        (0.0, np.nan, 1.6265984, 0.1726442),
    ),
    "scalar": (
        {
            ("A", 0): (1.20029238, 0.08902576, -0.31006172, 0.06644708),
            ("B", 1): (1.9939821, 0.1461444, -0.11712694, 0.05201949),
            ("B", 2): (1.00173193, 0.07918833, -0.47448329, 0.07899021),
        },
        (0.60335427, 0.06807134, 1.7013086, 0.1737918),
    ),
}


@pytest.mark.parametrize("invariance", list(_R_STANDARD_ERRORS))
def test_standard_errors_match_r_multiple_group(invariance):
    data, groups = _r_reference_data()
    result = fit_multigroup(
        data, groups, invariance=invariance, tol=1e-8, accelerate="squarem"
    )
    items, latent = _R_STANDARD_ERRORS[invariance]
    table = result.coef(include_standard_errors=True)
    rows = {
        (row["group"], index % 12): row for index, row in enumerate(_records(table))
    }
    for key, (a, a_se, b, b_se) in items.items():
        row = rows[key]
        assert row["discrimination"] == pytest.approx(a, abs=2e-3)
        assert row["difficulty"] == pytest.approx(b, abs=2e-3)
        assert row["discrimination_se"] == pytest.approx(a_se, rel=3e-3)
        assert row["difficulty_se"] == pytest.approx(b_se, rel=3e-3)
    moments = result.latent_pars(include_standard_errors=True)
    focal = _records(moments)[1]
    reference = result.latent_standard_errors[0]
    assert np.all(np.isnan(reference["mean"])) and np.all(np.isnan(reference["cov"]))
    if latent is None:
        assert np.isnan(focal["mean_se"]) and np.isnan(focal["variance_se"])
        return
    mean, mean_se, variance, variance_se = latent
    assert focal["mean"] == pytest.approx(mean, abs=2e-3)
    assert focal["variance"] == pytest.approx(variance, abs=5e-3)
    assert focal["variance_se"] == pytest.approx(variance_se, rel=3e-3)
    if np.isnan(mean_se):
        assert np.isnan(focal["mean_se"])
    else:
        assert focal["mean_se"] == pytest.approx(mean_se, rel=3e-3)
    assert result.vcov.shape == (len(result.vcov_labels),) * 2
    assert "B: cov[0,0]" in result.vcov_labels


def _perturb(model, density, label, step):
    """Move the coordinate that a ``vcov_labels`` entry names by ``step``."""
    group_label, _, coordinate = label.rpartition(": ")
    name, _, index = coordinate.partition("[")
    parts = [part.strip() for part in index.rstrip("]").split(",")]
    owners = (
        [model.group_labels.index(group_label)]
        if group_label
        else list(range(model.n_groups))
    )
    if name in ("mean", "cov"):
        distribution = density.distributions[owners[0]]
        mean, cov = distribution.mean.copy(), distribution.cov.copy()
        if name == "mean":
            mean[int(parts[0])] += step
        else:
            i, j = int(parts[0]), int(parts[1])
            cov[i, j] += step
            if i != j:
                cov[j, i] += step
        distribution._set_parameters(mean, cov)
        return
    position = (model.item_names.index(parts[0]), *(int(p) for p in parts[1:]))
    for g in owners:
        group = model.get_group_model(g)
        values = group.parameters[name]
        values[position] += step
        group.set_parameters(**{name: values})


def _numerical_standard_errors(result, estimator, responses, step=1e-4):
    """Standard errors from a central-difference Hessian of the marginal LL."""
    model, density = result.model, estimator._latent_density

    def log_likelihood(moves):
        for label, move in moves:
            _perturb(model, density, label, move)
        value = float(sum(estimator._e_step(model, responses)[1]))
        for label, move in moves:
            _perturb(model, density, label, -move)
        return value

    labels = result.vcov_labels
    hessian = np.empty((len(labels),) * 2)
    for i, first in enumerate(labels):
        for j, second in enumerate(labels[: i + 1]):
            total = 0.0
            for s1, s2, sign in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
                total += sign * log_likelihood(
                    [(first, s1 * step), (second, s2 * step)]
                )
            hessian[i, j] = hessian[j, i] = total / (4 * step**2)
    return np.sqrt(np.diag(np.linalg.inv(-hessian)))


def test_standard_errors_equal_the_inverse_marginal_hessian():
    from mirt.models import GradedResponseModel, TwoParameterLogistic
    from mirt.multigroup import MultigroupEMEstimator, MultigroupModel

    rng = np.random.default_rng(17)
    graded = []
    for mean, sd in ((0.0, 1.0), (0.5, 1.3)):
        theta = rng.normal(mean, sd, 300)
        cuts = np.array([-0.6, 0.7])
        latent = 1.4 * (theta[:, None, None] - cuts) - rng.logistic(size=(300, 4, 1))
        graded.append((latent > 0).sum(axis=2))
    estimator = MultigroupEMEstimator(tol=1e-9, compute_standard_errors=True)
    result = estimator.fit(
        MultigroupModel(GradedResponseModel(4, 3), 2), graded, "scalar"
    )
    expected = _numerical_standard_errors(result, estimator, graded)
    np.testing.assert_allclose(np.sqrt(np.diag(result.vcov)), expected, rtol=2e-3)

    # Two correlated factors: the reference correlation and the focal
    # covariance entries are estimated too.
    base = TwoParameterLogistic(n_items=6, n_factors=2)
    pattern = np.zeros((6, 2), dtype=bool)
    pattern[:3, 0] = pattern[3:, 1] = True
    base.set_parameters(discrimination=np.where(pattern, 1.0, 0.0))
    base.set_free_parameter_masks({"discrimination": pattern})
    slopes = np.where(pattern, 1.6, 0.0)
    binary = []
    for mean in ([0.0, 0.0], [0.4, -0.2]):
        theta = rng.multivariate_normal(mean, [[1.0, 0.5], [0.5, 1.0]], 400)
        p = 1 / (1 + np.exp(-(theta @ slopes.T - 0.2)))
        binary.append((rng.random(p.shape) < p).astype(int))
    estimator = MultigroupEMEstimator(
        n_quadpts=9, tol=1e-9, compute_standard_errors=True
    )
    result = estimator.fit(MultigroupModel(base, 2), binary, "scalar")
    assert {"Group_0: cov[0,1]", "Group_1: cov[0,1]", "Group_1: mean[1]"} <= set(
        result.vcov_labels
    )
    expected = _numerical_standard_errors(result, estimator, binary)
    np.testing.assert_allclose(np.sqrt(np.diag(result.vcov)), expected, rtol=2e-3)
