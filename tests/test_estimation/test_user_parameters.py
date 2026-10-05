"""User-controlled starting values and fixed parameters across estimators."""

import numpy as np
import pytest
from scipy.special import logsumexp

import mirt
import mirt.backends.rust.estimation as rust_estimation
from mirt.backends.rust._helpers import RUST_AVAILABLE
from mirt.estimation.bl import BLEstimator
from mirt.estimation.em import EMEstimator
from mirt.estimation.gvem import GVEMEstimator
from mirt.estimation.mcem import MCEMEstimator, QMCEMEstimator, StochasticEMEstimator
from mirt.estimation.mcmc import GibbsSampler, MHRMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.sparse_bayesian import SparseBayesianEstimator
from mirt.estimation.weighted import WeightedEMEstimator
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import ThreeParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel
from mirt.utils.starting import gen_random_pars

_N_ITEMS = 8


@pytest.fixture(scope="module")
def responses():
    return mirt.simdata("2PL", n_persons=300, n_items=_N_ITEMS, seed=3)


def _fixed_model():
    model = TwoParameterLogistic(_N_ITEMS)
    discrimination = model.parameters["discrimination"]
    discrimination[0] = 2.5
    model.set_parameters(discrimination=discrimination)
    mask = np.ones(_N_ITEMS, dtype=bool)
    mask[0] = False
    return model.set_free_parameter_masks({"discrimination": mask})


def _marginal_log_likelihood(model, responses, n_quadpts=21):
    quadrature = GaussHermiteQuadrature(n_quadpts)
    log_joint = model.log_likelihood_batch(responses, quadrature.nodes) + np.log(
        quadrature.weights
    )
    return float(np.sum(logsumexp(log_joint, axis=1)))


@pytest.mark.parametrize(
    "estimator",
    [
        EMEstimator(max_iter=20),
        BLEstimator(),
        WeightedEMEstimator(max_iter=20),
        MCEMEstimator(n_samples=50, max_iter=3, seed=1),
        QMCEMEstimator(n_samples=64, max_iter=3, seed=1),
        StochasticEMEstimator(max_iter=3, seed=1),
    ],
    ids=lambda estimator: type(estimator).__name__,
)
def test_fixed_coordinates_survive_fitting(estimator, responses):
    model = _fixed_model()
    result = estimator.fit(model, responses)
    discrimination = result.model.parameters["discrimination"]
    assert discrimination[0] == 2.5
    assert not np.allclose(discrimination[1:], 1.0)


@pytest.mark.parametrize(
    "make_estimator",
    [
        lambda: MHRMEstimator(n_cycles=5, burnin=2, use_rust=True),
        lambda: MHRMEstimator(n_cycles=5, burnin=2, use_rust=False),
        lambda: GibbsSampler(n_iter=5, burnin=2, use_rust=True),
        lambda: GibbsSampler(n_iter=5, burnin=2, use_rust=False),
        lambda: GVEMEstimator(max_iter=5),
        lambda: SparseBayesianEstimator(k_max=1, max_iter=5),
    ],
    ids=["mhrm-native", "mhrm-numpy", "gibbs-native", "gibbs-numpy", "gvem", "ssl"],
)
def test_estimators_that_move_fixed_coordinates_reject_masks(make_estimator, responses):
    model = _fixed_model()
    before = model.parameters
    with pytest.raises(MirtValidationError, match="set_free_parameter_masks"):
        make_estimator().fit(model, responses)
    for name, values in before.items():
        np.testing.assert_array_equal(model.parameters[name], values)


def test_start_model_begins_at_supplied_values(responses):
    model = TwoParameterLogistic(_N_ITEMS)
    model.set_parameters(discrimination=np.full(_N_ITEMS, 1.7))
    expected = _marginal_log_likelihood(model, responses)
    estimator = EMEstimator(max_iter=1, compute_standard_errors=False)
    estimator.fit(model, responses, start="model")
    assert estimator.convergence_history[0] == pytest.approx(expected, rel=1e-12)


def test_default_start_resets_unfitted_and_warm_starts_fitted_models(responses):
    model = TwoParameterLogistic(_N_ITEMS)
    model.set_parameters(discrimination=np.full(_N_ITEMS, 1.7))
    estimator = EMEstimator(max_iter=1, compute_standard_errors=False)
    estimator.fit(model, responses)
    defaults = _marginal_log_likelihood(TwoParameterLogistic(_N_ITEMS), responses)
    assert estimator.convergence_history[0] == pytest.approx(defaults, rel=1e-12)

    fitted = EMEstimator(compute_standard_errors=False).fit(
        TwoParameterLogistic(_N_ITEMS), responses
    )
    warm = _marginal_log_likelihood(fitted.model, responses)
    estimator.fit(fitted.model, responses)
    assert estimator.convergence_history[0] == pytest.approx(warm, rel=1e-12)


def test_start_mapping_overrides_defaults_and_sets_fixed_values(responses):
    model = TwoParameterLogistic(_N_ITEMS)
    model.set_parameters(difficulty=np.full(_N_ITEMS, 3.0))
    mask = np.ones(_N_ITEMS, dtype=bool)
    mask[1] = False
    model.set_free_parameter_masks({"discrimination": mask})
    start = np.linspace(0.5, 2.0, _N_ITEMS)
    estimator = EMEstimator(max_iter=1, compute_standard_errors=False)
    result = estimator.fit(model, responses, start={"discrimination": start})

    reference = TwoParameterLogistic(_N_ITEMS).set_parameters(discrimination=start)
    expected = _marginal_log_likelihood(reference, responses)
    assert estimator.convergence_history[0] == pytest.approx(expected, rel=1e-12)
    assert result.model.parameters["discrimination"][1] == start[1]


@pytest.mark.parametrize(
    ("start", "message"),
    [
        ("random", "start must be"),
        (3, "start must be"),
        ({"slope": np.ones(_N_ITEMS)}, "Unknown parameter"),
        ({"discrimination": np.ones(3)}, "must have shape"),
        ({"discrimination": np.full(_N_ITEMS, np.nan)}, "must be finite"),
        ({"discrimination": ["a"] * _N_ITEMS}, "must be numeric"),
    ],
)
def test_invalid_start_is_rejected_without_changing_model(responses, start, message):
    model = TwoParameterLogistic(_N_ITEMS)
    model.set_parameters(discrimination=np.full(_N_ITEMS, 1.7))
    with pytest.raises(MirtValidationError, match=message):
        EMEstimator(max_iter=1).fit(model, responses, start=start)
    np.testing.assert_array_equal(model.parameters["discrimination"], 1.7)


@pytest.mark.parametrize(
    "estimator",
    [
        GVEMEstimator(max_iter=1),
        MCEMEstimator(n_samples=50, max_iter=1, seed=2),
        WeightedEMEstimator(max_iter=1),
    ],
    ids=lambda estimator: type(estimator).__name__,
)
def test_other_estimators_honor_start(estimator, responses, monkeypatch):
    seen = []

    def record(model, *args, **kwargs):
        seen.append(model.parameters["discrimination"].copy())
        raise RuntimeError("stop after initialization")

    hook = {
        GVEMEstimator: "_convert_to_slope_intercept",
        MCEMEstimator: "_e_step_and_marginal_ll",
        WeightedEMEstimator: "_e_step_weighted",
    }[type(estimator)]
    monkeypatch.setattr(estimator, hook, record)
    start = np.linspace(0.6, 1.8, _N_ITEMS)
    model = TwoParameterLogistic(_N_ITEMS)
    with pytest.raises(RuntimeError, match="stop after initialization"):
        estimator.fit(model, responses, start={"discrimination": start})
    model.set_parameters(discrimination=np.full(_N_ITEMS, 1.3))
    with pytest.raises(RuntimeError, match="stop after initialization"):
        estimator.fit(model, responses, start="model")
    with pytest.raises(RuntimeError, match="stop after initialization"):
        estimator.fit(model, responses)
    np.testing.assert_array_equal(seen[0], start)
    np.testing.assert_array_equal(seen[1], 1.3)
    np.testing.assert_array_equal(seen[2], 1.0)


def test_fit_mirt_start_values_and_fixed_hold_coordinates(responses, monkeypatch):
    def native_fit(*args, **kwargs):
        raise AssertionError("the native 2PL fast path ignores user values")

    monkeypatch.setattr(rust_estimation, "_em_fit_2pl_prepared", native_fit)
    start = np.full(_N_ITEMS, 1.2)
    start[0] = 2.25
    fixed = np.zeros(_N_ITEMS, dtype=bool)
    fixed[0] = True
    result = mirt.fit_mirt(
        responses,
        start_values={"discrimination": start},
        fixed={"discrimination": fixed, "difficulty": False},
    )
    discrimination = result.model.parameters["discrimination"]
    assert discrimination[0] == 2.25
    assert result.n_parameters == 2 * _N_ITEMS - 1
    assert result.standard_errors["discrimination"][0] == 0.0
    assert np.all(result.standard_errors["discrimination"][1:] > 0.0)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="native 2PL EM is unavailable")
def test_fit_mirt_keeps_the_native_path_without_effective_restrictions(
    responses, monkeypatch
):
    calls = []
    native = rust_estimation._em_fit_2pl_prepared

    def record(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(rust_estimation, "_em_fit_2pl_prepared", record)
    mirt.fit_mirt(
        responses,
        fixed={"difficulty": False},
        priors={},
        compute_standard_errors=False,
    )
    assert calls == [True]


def test_fit_mirt_scalar_fixed_mask_fixes_a_whole_parameter(responses):
    guessing = np.full(_N_ITEMS, 0.15)
    data = mirt.simdata("3PL", n_persons=300, n_items=_N_ITEMS, seed=5)
    result = mirt.fit_mirt(
        data,
        model="3PL",
        start_values={"guessing": guessing},
        fixed={"guessing": True},
        compute_standard_errors=False,
    )
    np.testing.assert_array_equal(result.model.parameters["guessing"], guessing)
    assert result.n_parameters == 2 * _N_ITEMS


@pytest.mark.parametrize(
    ("fixed", "message"),
    [
        ([True], "must map parameter names"),
        ({"slope": True}, "Unknown parameter"),
        ({"discrimination": 1}, "must be Boolean"),
        ({"discrimination": np.ones(3, dtype=bool)}, "must be a scalar or have shape"),
    ],
)
def test_fit_mirt_rejects_invalid_fixed_masks(responses, fixed, message):
    with pytest.raises(MirtValidationError, match=message):
        mirt.fit_mirt(responses, fixed=fixed)


def test_fit_mirt_samplers_reject_fixed_and_honor_start_values(responses, monkeypatch):
    with pytest.raises(MirtValidationError, match="set_free_parameter_masks"):
        mirt.fit_mirt(
            responses, estimation="MHRM", max_iter=5, fixed={"discrimination": True}
        )

    seen = {}
    original = MHRMEstimator.fit

    def record(self, model, data, **kwargs):
        seen["discrimination"] = model.parameters["discrimination"].copy()
        seen["use_rust"] = self.use_rust
        return original(self, model, data, **kwargs)

    monkeypatch.setattr(MHRMEstimator, "fit", record)
    start = np.linspace(0.8, 1.6, _N_ITEMS)
    mirt.fit_mirt(
        responses,
        estimation="MHRM",
        max_iter=4,
        start_values={"discrimination": start},
    )
    np.testing.assert_array_equal(seen["discrimination"], start)
    assert seen["use_rust"] is False


def test_gen_random_pars_keeps_user_fixed_coordinates():
    model = ThreeParameterLogistic(4)
    model.set_parameters(guessing=np.array([0.05, 0.2, 0.2, 0.2]))
    mask = np.array([False, True, True, True])
    model.set_free_parameter_masks({"guessing": mask})
    for values in gen_random_pars(model, n_sets=3, seed=4):
        assert values["guessing"][0] == 0.05
        assert np.all(values["guessing"][1:] != 0.2)


def test_gen_random_pars_keeps_graded_thresholds_ordered_around_fixed_ones():
    model = GradedResponseModel(3, n_categories=4)
    current = np.array([[-1.0, 1.9, 2.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    model.set_parameters(thresholds=current)
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 1] = False
    model.set_free_parameter_masks({"thresholds": mask})
    for values in gen_random_pars(model, n_sets=5, seed=1):
        thresholds = values["thresholds"]
        assert thresholds[0, 1] == 1.9
        assert np.all(np.diff(thresholds, axis=1) > 0.0)
