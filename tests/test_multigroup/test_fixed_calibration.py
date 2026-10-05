"""Independent checks for fixed blocks and constrained multigroup estimation."""

from types import MethodType

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp

from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import (
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)
from mirt.multigroup.estimator import MultigroupEMEstimator
from mirt.multigroup.latent import MultigroupLatentDensity
from mirt.multigroup.model import MultigroupModel


@pytest.mark.parametrize("invariance", ["configural", "scalar"])
def test_fixed_blocks_hold_before_every_e_step_and_exclude_fit_parameters(invariance):
    rng = np.random.default_rng(471)
    responses = [rng.integers(0, 2, (120, 4)) for _ in range(2)]
    fixed = {"discrimination": {0: 1.37}, "difficulty": {0: -0.63}}

    class ObservedEstimator(MultigroupEMEstimator):
        checks = 0

        def _e_step(self, model, responses):
            for group in model.group_models:
                assert group.parameters["discrimination"][0] == 1.37
                assert group.parameters["difficulty"][0] == -0.63
            self.checks += 1
            return super()._e_step(model, responses)

    model = MultigroupModel(TwoParameterLogistic(4), 2)
    estimator = ObservedEstimator(n_quadpts=15, max_iter=5)
    fit = estimator.fit(model, responses, invariance, fixed_parameters=fixed)
    assert estimator.checks >= 5
    expected_item_parameters = 12 if invariance == "configural" else 6
    assert model.n_parameters == expected_item_parameters
    assert fit.n_parameters == expected_item_parameters + 2
    assert fit.aic == -2 * fit.log_likelihood + 2 * fit.n_parameters
    for name in fixed:
        assert not model.effective_free_parameter_masks(0)[name][0]


def test_fixed_registry_is_copied_atomic_and_survives_synchronization():
    model = MultigroupModel(GradedResponseModel(2, [2, 3]), 2)
    values = np.array([-0.4, 0.8])
    model.fix_item_parameters({"thresholds": {1: values}})
    values[:] = 7
    exported = model.fixed_item_parameters
    exported["thresholds"][1][:] = 9
    for group in model.group_models:
        np.testing.assert_array_equal(group.parameters["thresholds"][1], [-0.4, 0.8])
    before = [group.parameters for group in model.group_models]
    with pytest.raises(ValueError, match="shape"):
        model.fix_item_parameters({"discrimination": {0: 1.8}, "thresholds": {1: 2.0}})
    for group, original in zip(model.group_models, before, strict=True):
        for name, value in original.items():
            np.testing.assert_array_equal(group.parameters[name], value)
    model.set_shared_parameter("thresholds")
    model.get_group_model(1)._parameters["thresholds"][1] = [-4, 4]
    model.synchronize_shared_parameters()
    for group in model.group_models:
        np.testing.assert_array_equal(group.parameters["thresholds"][1], [-0.4, 0.8])


def test_fixed_validation_preserves_instance_setters_and_is_atomic_across_groups():
    model = MultigroupModel(TwoParameterLogistic(2), 2)

    def constrained_setter(self, **parameters):
        difficulty = parameters.get("difficulty", self.parameters["difficulty"])
        if np.any(difficulty > self.difficulty_ceiling):
            raise ValueError("difficulty exceeds this group's ceiling")
        return TwoParameterLogistic.set_parameters(self, **parameters)

    for group_idx, group in enumerate(model.group_models):
        group.difficulty_ceiling = 2.0 if group_idx == 0 else 1.0
        group.set_parameters = MethodType(constrained_setter, group)

    model.fix_item_parameters({"difficulty": {0: 0.2}})
    before = [group.parameters for group in model.group_models]
    with pytest.raises(ValueError, match="this group's ceiling"):
        model.fix_item_parameters({"discrimination": {0: 1.8}, "difficulty": {0: 1.5}})
    for group, original in zip(model.group_models, before, strict=True):
        for name, values in original.items():
            np.testing.assert_array_equal(group.parameters[name], values)
    assert model.fixed_item_parameters == {"difficulty": {0: np.array(0.2)}}
    for group_idx in range(model.n_groups):
        masks = model.effective_free_parameter_masks(group_idx)
        assert not masks["difficulty"][0]
        assert masks["discrimination"][0]


@pytest.mark.parametrize(
    "base",
    [
        OneParameterLogistic(3),
        PartialCreditModel(3, [2, 3, 4]),
        NominalResponseModel(3, [2, 3, 4]),
    ],
)
def test_structural_fixed_coordinates_never_move_in_m_step(base):
    model = MultigroupModel(base, 2)
    before = [group.parameters for group in model.group_models]
    rng = np.random.default_rng(802)
    categories = base._n_categories if base.is_polytomous else [2] * base.n_items
    responses = [
        np.column_stack([rng.integers(0, count, 40) for count in categories])
        for _ in range(2)
    ]
    weights = [rng.dirichlet(np.ones(5), 40) for _ in range(2)]
    quad = np.linspace(-2, 2, 5)[:, None]
    estimator = MultigroupEMEstimator(item_optim_ftol=1e-10)
    for item in range(base.n_items):
        estimator._optimize_item(model, item, responses, weights, quad)
    moved = False
    for group, original in zip(model.group_models, before, strict=True):
        for name, mask in group.free_parameter_masks.items():
            np.testing.assert_array_equal(
                group.parameters[name][~mask], original[name][~mask]
            )
            moved |= bool(np.any(group.parameters[name][mask] != original[name][mask]))
    assert moved


def test_fixed_slope_item_difficulty_matches_independent_conditional_likelihood():
    model = MultigroupModel(TwoParameterLogistic(1), 2)
    model.fix_item_parameters({"discrimination": {0: 1.73}})
    responses = np.array([[0], [1], [1], [-1], [0], [1]])
    posterior = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.3, 0.5],
            [0.05, 0.2, 0.75],
            [0.2, 0.6, 0.2],
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
        ]
    )
    nodes = np.array([[-1.5], [0.2], [1.8]])

    def loss(difficulty):
        probabilities = expit(1.73 * (nodes[:, 0] - difficulty))
        return -sum(
            float(
                posterior[i]
                @ np.log(probabilities if response == 1 else 1 - probabilities)
            )
            for i, response in enumerate(responses[:, 0])
            if response >= 0
        )

    optimum = minimize_scalar(
        loss, bounds=(-6, 6), method="bounded", options={"xatol": 1e-12}
    )
    assert optimum.success
    estimator = MultigroupEMEstimator(item_optim_ftol=1e-12)
    estimator._optimize_item(
        model, 0, [responses, responses], [posterior, posterior], nodes
    )
    for group in model.group_models:
        assert group.parameters["difficulty"][0] == pytest.approx(optimum.x, abs=1e-6)
        assert group.parameters["discrimination"][0] == 1.73


def test_nominal_multidimensional_parameter_rows_are_flattened_without_freeing_reference():
    base = NominalResponseModel(2, [2, 3], n_factors=2)
    model = MultigroupModel(base, 2)
    rng = np.random.default_rng(876)
    responses = np.column_stack((rng.integers(0, 2, 50), rng.integers(0, 3, 50)))
    weights = rng.dirichlet(np.ones(9), 50)
    nodes = GaussHermiteQuadrature(3, 2).nodes
    estimator = MultigroupEMEstimator()
    estimator._optimize_item(model, 1, [responses] * 2, [weights] * 2, nodes)
    fitted = model.get_group_model(1)
    np.testing.assert_array_equal(fitted.parameters["slopes"][1, 0], [0, 0])
    assert np.any(
        fitted.parameters["slopes"][1, 1:] != base.parameters["slopes"][1, 1:]
    )


def test_ordered_density_update_recovers_independent_discrete_gaussian_optima():
    estimator = MultigroupEMEstimator(n_quadpts=31)
    estimator._quadrature = GaussHermiteQuadrature(31, 1)
    estimator._latent_density = MultigroupLatentDensity(3, 1, reference_group=1)
    nodes = estimator._quadrature.nodes[:, 0]
    log_base = np.log(estimator._quadrature.weights) + nodes**2 / 2
    targets = [(-0.6, 0.7), (0, 1), (0.8, 1.3)]
    posterior = []
    for mean, variance in targets:
        log_mass = log_base - (nodes - mean) ** 2 / (2 * variance)
        posterior.append(np.tile(np.exp(log_mass - logsumexp(log_mass)), (100, 1)))
    estimator._update_ordered_latent_density(posterior, (0, 1, 2))
    for distribution, (mean, variance) in zip(
        estimator._latent_density.distributions, targets, strict=True
    ):
        assert distribution.mean[0] == pytest.approx(mean, abs=2e-6)
        assert distribution.cov[0, 0] == pytest.approx(variance, abs=3e-6)


def test_grm_m_step_retains_ordered_thresholds_and_zero_padding():
    model = MultigroupModel(GradedResponseModel(2, [2, 4]), 2)
    rng = np.random.default_rng(917)
    responses = np.column_stack((rng.integers(0, 2, 90), rng.integers(0, 4, 90)))
    responses[:60, 1] = 0
    weights = rng.dirichlet(np.ones(9), 90)
    nodes = np.linspace(-4, 4, 9)[:, None]
    estimator = MultigroupEMEstimator(item_optim_maxiter=100)
    for item in range(2):
        estimator._optimize_item(model, item, [responses] * 2, [weights] * 2, nodes)
    group = model.get_group_model(1)
    assert np.all(np.diff(group.parameters["thresholds"][1]) >= 0)
    np.testing.assert_array_equal(group.parameters["thresholds"][0, 1:], [0, 0])
    assert np.all(group.probability(nodes) >= 0)


@pytest.mark.parametrize("failure", ["raise", "nan", "worse"])
def test_numerical_optimizer_failure_restores_custom_model_state(monkeypatch, failure):
    from types import MethodType, SimpleNamespace

    import mirt.multigroup.estimator as estimator_module

    model = MultigroupModel(TwoParameterLogistic(1), 2)
    group = model.get_group_model(1)
    original_probability = TwoParameterLogistic.probability

    def custom_probability(self, theta, item_idx=None):
        return original_probability(self, theta, item_idx)

    group.probability = MethodType(custom_probability, group)
    # The custom curve sends the item, in both groups, to the per-parameter
    # path, whose trials go through the models' setters.
    before = [member.parameters for member in model.group_models]
    responses = np.array([[0], [1], [0], [1]])
    posterior = np.full((4, 5), 0.2)
    nodes = np.linspace(-2, 2, 5)[:, None]

    def failed_optimizer(objective, x0, **kwargs):
        objective(np.array([2.7]))
        if failure == "raise":
            raise RuntimeError("trial callback failure")
        return SimpleNamespace(x=np.array([np.nan if failure == "nan" else 6.0]))

    monkeypatch.setattr(estimator_module, "minimize", failed_optimizer)
    estimator = MultigroupEMEstimator()
    if failure == "raise":
        with pytest.raises(RuntimeError, match="trial callback"):
            estimator._optimize_item(model, 0, [responses] * 2, [posterior] * 2, nodes)
    else:
        estimator._optimize_item(model, 0, [responses] * 2, [posterior] * 2, nodes)
    for member, parameters in zip(model.group_models, before, strict=True):
        for name, value in parameters.items():
            np.testing.assert_array_equal(member.parameters[name], value)


@pytest.mark.parametrize("order", [[0], [0, 0], [False, 1], [0, 2], "01"])
def test_invalid_mean_orders_are_rejected_before_invariance_changes(order):
    model = MultigroupModel(TwoParameterLogistic(2), 2)
    responses = [np.array([[0, 1], [1, 0]])] * 2
    with pytest.raises(ValueError, match="mean_order"):
        MultigroupEMEstimator().fit(model, responses, "scalar", mean_order=order)
    assert not model.is_item_parameter_shared("difficulty", 0)


@pytest.mark.parametrize(
    "responses",
    [
        np.zeros(2),
        np.zeros((0, 2)),
        np.array([[0, 0.5]]),
        np.array([[0, np.inf]]),
        np.array([[0, 2]]),
        np.array([[-1, -1]]),
    ],
)
def test_invalid_responses_fail_before_fitting_or_changing_constraints(responses):
    model = MultigroupModel(TwoParameterLogistic(2), 2)
    before = [group.parameters for group in model.group_models]
    with pytest.raises(ValueError):
        MultigroupEMEstimator().fit(model, [responses, np.array([[0, 1]])], "scalar")
    assert not model.is_item_parameter_shared("difficulty", 0)
    for group, original in zip(model.group_models, before, strict=True):
        for name, value in original.items():
            np.testing.assert_array_equal(group.parameters[name], value)


@pytest.mark.parametrize(
    "base, fixed",
    [
        (OneParameterLogistic(3), {"difficulty": {0: 0.3}}),
        (PartialCreditModel(3, [2, 3, 4]), {"steps": {1: np.array([-1.0, 1.0, 0.0])}}),
    ],
)
def test_fixed_parameters_work_with_rasch_and_partial_credit_setter_contracts(
    base, fixed
):
    model = MultigroupModel(base, 2)
    model.fix_item_parameters(fixed)
    model.fix_item_parameters({"discrimination": {0: 1.0}})
    rng = np.random.default_rng(928)
    categories = base._n_categories if base.is_polytomous else [2] * base.n_items
    responses = [
        np.column_stack([rng.integers(0, count, 40) for count in categories])
        for _ in range(2)
    ]
    result = MultigroupEMEstimator(n_quadpts=11, max_iter=4).fit(
        model, responses, "scalar"
    )
    assert np.isfinite(result.log_likelihood)
    for group in model.group_models:
        np.testing.assert_array_equal(group.parameters["discrimination"], np.ones(3))
        for name, items in fixed.items():
            for item, value in items.items():
                np.testing.assert_array_equal(group.parameters[name][item], value)


def test_ordered_density_handles_conflicting_means_by_constrained_likelihood():
    estimator = MultigroupEMEstimator(n_quadpts=31)
    estimator._quadrature = GaussHermiteQuadrature(31, 1)
    estimator._latent_density = MultigroupLatentDensity(3, 1, reference_group=1)
    nodes = estimator._quadrature.nodes[:, 0]
    log_base = np.log(estimator._quadrature.weights) + nodes**2 / 2
    posterior = []
    for mean, variance in [(0.7, 0.6), (0, 1), (-0.5, 1.4)]:
        log_mass = log_base - (nodes - mean) ** 2 / (2 * variance)
        posterior.append(np.tile(np.exp(log_mass - logsumexp(log_mass)), (100, 1)))
    estimator._update_ordered_latent_density(posterior, (0, 1, 2))
    for index in [0, 2]:
        counts = posterior[index].sum(axis=0)

        def fixed_mean_loss(log_variance):
            log_mass = log_base - nodes**2 / (2 * np.exp(log_variance))
            return -float(counts @ (log_mass - logsumexp(log_mass)))

        optimum = minimize_scalar(
            fixed_mean_loss,
            bounds=(-14, 14),
            method="bounded",
            options={"xatol": 1e-12},
        )
        assert optimum.success
        fitted = estimator._latent_density.distributions[index]
        assert fitted.mean[0] == pytest.approx(0, abs=1e-8)
        # The constrained solver stops on Q rather than coordinate change;
        # require statistical agreement across BLAS summation orders.
        assert fitted.cov[0, 0] == pytest.approx(np.exp(optimum.x), abs=1e-5)


def test_multigroup_partial_grm_thresholds_respect_fixed_neighbors():
    base = GradedResponseModel(1, 4)
    base.set_parameters(thresholds=np.array([[-1.0, 0.0, 1.0]]))
    base.set_free_parameter_masks({"thresholds": np.array([[False, False, True]])})
    model = MultigroupModel(base, 2)
    responses = np.array([[0]] * 90 + [[1]] * 5 + [[3]] * 5)
    nodes = np.linspace(-4, 4, 9)[:, None]
    posterior = np.full((100, 9), 1 / 9)
    MultigroupEMEstimator(item_optim_maxiter=100)._optimize_item(
        model, 0, [responses] * 2, [posterior] * 2, nodes
    )
    group = model.get_group_model(1)
    np.testing.assert_array_equal(group.parameters["thresholds"][0, :2], [-1, 0])
    assert group.parameters["thresholds"][0, 2] >= 0
    assert np.all(group.probability(nodes) >= 0)


def test_initialization_preserves_explicit_fixed_coordinates():
    base = TwoParameterLogistic(2)
    base.set_parameters(
        discrimination=np.array([1.7, 1.0]), difficulty=np.array([2.0, 0.0])
    )
    base.set_free_parameter_masks(
        {
            "discrimination": np.array([False, True]),
            "difficulty": np.array([False, True]),
        }
    )
    model = MultigroupModel(base, 2)

    class CheckedEstimator(MultigroupEMEstimator):
        def _e_step(self, model, responses):
            for group in model.group_models:
                assert group.parameters["discrimination"][0] == 1.7
                assert group.parameters["difficulty"][0] == 2.0
            return super()._e_step(model, responses)

    rng = np.random.default_rng(825)
    result = CheckedEstimator(max_iter=3).fit(
        model, [rng.integers(0, 2, (30, 2)) for _ in range(2)]
    )
    assert np.isfinite(result.log_likelihood)


def test_warm_refit_establishes_shared_constraints_before_first_e_step():
    model = MultigroupModel(TwoParameterLogistic(2), 2)
    for group, difficulty in zip(model.group_models, [0.2, 1.2], strict=True):
        group.set_parameters(difficulty=np.array([difficulty, difficulty - 0.3]))
        group._is_fitted = True

    class CheckedEstimator(MultigroupEMEstimator):
        def _e_step(self, model, responses):
            np.testing.assert_array_equal(
                model.get_group_model(0).parameters["difficulty"],
                model.get_group_model(1).parameters["difficulty"],
            )
            return super()._e_step(model, responses)

    responses = [np.array([[0, 1], [1, 0], [1, 1]])] * 2
    CheckedEstimator(max_iter=2).fit(model, responses, "scalar")


def test_invalid_reference_is_rejected_before_changing_constraints():
    model = MultigroupModel(TwoParameterLogistic(2), 2)
    with pytest.raises(ValueError, match="reference_group"):
        MultigroupEMEstimator().fit(
            model, [np.array([[0, 1]])] * 2, "scalar", reference_group=3
        )
    assert not model.is_item_parameter_shared("difficulty", 0)


def test_shared_coordinate_fixed_in_one_group_is_known_everywhere():
    model = MultigroupModel(TwoParameterLogistic(2), 2)
    reference = model.get_group_model(0)
    reference.set_parameters(difficulty=np.array([0.37, -0.5]))
    reference.set_free_parameter_masks({"difficulty": np.array([False, True])})
    model.set_shared_parameter("difficulty")
    assert model.n_parameters == 5  # four group-specific slopes, one shared difficulty
    model.synchronize_shared_parameters()
    for group in model.group_models:
        assert group.parameters["difficulty"][0] == 0.37
    for index in range(2):
        np.testing.assert_array_equal(
            model.effective_free_parameter_masks(index)["difficulty"], [False, True]
        )
    rng = np.random.default_rng(965)
    responses = [rng.integers(0, 2, (40, 2)) for _ in range(2)]
    posterior = [rng.dirichlet(np.ones(7), 40) for _ in range(2)]
    MultigroupEMEstimator()._optimize_item(
        model, 0, responses, posterior, np.linspace(-3, 3, 7)[:, None]
    )
    for group in model.group_models:
        assert group.parameters["difficulty"][0] == 0.37
    model.copy_shared_to_all(source_group=1)
    for group in model.group_models:
        assert group.parameters["difficulty"][0] == 0.37


def test_conflicting_shared_fixed_values_are_rejected_atomically():
    model = MultigroupModel(TwoParameterLogistic(2), 2)
    for group, value in zip(model.group_models, [0.37, -0.29], strict=True):
        group.set_parameters(difficulty=np.array([value, 0.7]))
        group.set_free_parameter_masks({"difficulty": np.array([False, True])})
    model.set_shared_parameter("difficulty")
    before = [group.parameters for group in model.group_models]
    with pytest.raises(ValueError, match="incompatible fixed"):
        model.synchronize_shared_parameters()
    for group, original in zip(model.group_models, before, strict=True):
        for name, values in original.items():
            np.testing.assert_array_equal(group.parameters[name], values)
