"""Independent posterior likelihood oracles, batching, and model hooks."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit

import mirt
import mirt.diagnostics.bayesian as bayesian
from mirt.backends.rust import diagnostics as backend
from mirt.models.base import BaseItemModel
from mirt.models.dichotomous import ThreeParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


@pytest.fixture(params=["numpy", "rust"])
def selected_backend(request):
    if request.param == "rust" and not mirt.is_rust_available():
        pytest.skip("Rust extension is unavailable")
    previous = mirt.get_backend()
    mirt.set_backend(request.param)
    yield request.param
    mirt.set_backend(previous)


def _case(kind):
    counts = [3, 4, 2]
    responses = np.array([[0, 1, -1], [1, -1, 0], [0, 0, 1], [-1, -1, -1], [1, 1, 1]])
    theta = np.array([[-1.2, -0.4, 0.2, 0.8, 1.6], [-0.8, -0.1, 0.5, 1.3, 1.9]])
    a = np.array([[0.7, 1.1, 1.6], [0.9, 1.4, 1.2]])
    chains = {"discrimination": a, "theta": theta}
    if kind == "GRM":
        model = GradedResponseModel(3, counts)
        thresholds = np.array(
            [
                [[-1.1, 0.9, 0], [-1.3, 0.1, 1.4], [0.4, 0, 0]],
                [[-0.9, 1.2, 0], [-1.0, 0.3, 1.7], [0.7, 0, 0]],
            ]
        )
        chains["thresholds"] = thresholds
        responses[0, 1] = 3
        responses[2, 0] = 2
        oracle = np.zeros((2, *responses.shape))
        for s, person, item in np.ndindex(oracle.shape):
            k = responses[person, item]
            if k < 0:
                continue
            cumulative = expit(
                a[s, item]
                * (theta[s, person] - thresholds[s, item, : counts[item] - 1])
            )
            probabilities = np.diff(np.r_[1, cumulative, 0]) * -1
            oracle[s, person, item] = np.log(probabilities[k])
    else:
        model = TwoParameterLogistic(3) if kind == "2PL" else ThreeParameterLogistic(3)
        b = np.array([[-0.7, 0.2, 1.1], [-0.4, 0.5, 1.3]])
        chains["difficulty"] = b
        p = expit(a[:, None, :] * (theta[:, :, None] - b[:, None, :]))
        if kind == "3PL":
            c = np.array([[0.1, 0.2, 0.15], [0.2, 0.1, 0.25]])
            chains["guessing"] = c
            p = c[:, None, :] + (1 - c[:, None, :]) * p
        oracle = np.where(responses[None] == 1, np.log(p), np.log1p(-p))
        oracle[:, responses < 0] = 0
    return model, responses, chains, oracle


def _aggregate(oracle, responses, by):
    if by == "person":
        return oracle.sum(axis=2)
    if by == "observation":
        return oracle.reshape(oracle.shape[0], -1)
    return oracle[:, responses >= 0]


@pytest.mark.parametrize("kind", ["2PL", "3PL", "GRM"])
@pytest.mark.parametrize("by", ["person", "observation", "observed"])
@pytest.mark.parametrize("batch_size", [None, 1, 2, 20])
def test_streamed_pointwise_matches_independent_equations(
    kind, by, batch_size, selected_backend
):
    model, responses, chains, oracle = _case(kind)
    original = model.parameters
    chains = {name: values.copy() for name, values in chains.items()}
    for values in chains.values():
        values.setflags(write=False)
    actual = bayesian.compute_pointwise_log_lik(
        model, responses, chains, by, batch_size=batch_size
    )
    np.testing.assert_allclose(actual, _aggregate(oracle, responses, by), atol=2e-14)
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize("override", ["instance", "class", "inherited", "theta"])
def test_pointwise_respects_original_and_inherited_curve_hooks(
    override, selected_backend, monkeypatch
):
    model, responses, chains, _ = _case("2PL")
    calls = []

    def curve(self, theta, item_idx=None):
        calls.append(theta.copy())
        return np.full((len(theta), self.n_items), 0.8)

    if override == "instance":
        monkeypatch.setattr(model, "probability", MethodType(curve, model))
    elif override == "class":
        monkeypatch.setattr(TwoParameterLogistic, "probability", curve)
    elif override == "inherited":
        original = BaseItemModel._ensure_theta_2d

        def shifted(self, theta):
            calls.append(theta.copy())
            return original(self, theta) + 1.25

        monkeypatch.setattr(BaseItemModel, "_ensure_theta_2d", shifted)
    else:
        original = model._ensure_theta_2d

        def shifted(theta):
            calls.append(theta.copy())
            return original(theta) + 1.25

        monkeypatch.setattr(model, "_ensure_theta_2d", shifted)

    actual = bayesian.compute_pointwise_log_lik(
        model, responses, chains, by="observation"
    )
    if override in ("instance", "class"):
        expected = np.where(responses == 1, np.log(0.8), np.log(0.2))
        expected[responses < 0] = 0
        expected = np.tile(expected.ravel(), (2, 1))
    else:
        p = expit(
            chains["discrimination"][:, None, :]
            * (chains["theta"][:, :, None] + 1.25 - chains["difficulty"][:, None, :])
        )
        expected = np.where(responses[None] == 1, np.log(p), np.log1p(-p))
        expected[:, responses < 0] = 0
        expected = expected.reshape(2, -1)
    np.testing.assert_allclose(actual, expected)
    assert len(calls) == 2
    assert all(len(theta) == len(responses) for theta in calls)


@pytest.mark.parametrize("by", ["person", "observation", "observed"])
def test_default_ordinal_batches_bound_category_scratch(by, monkeypatch):
    model, responses, chains, oracle = _case("GRM")
    calls = []
    original = bayesian._pointwise_log_likelihood

    def tracked(model, responses, theta):
        calls.append(len(responses))
        return original(model, responses, theta)

    monkeypatch.setattr(bayesian, "_LOG_LIKELIHOOD_CHUNK_ELEMENTS", 24)
    monkeypatch.setattr(bayesian, "_pointwise_log_likelihood", tracked)
    actual = bayesian.compute_pointwise_log_lik(model, responses, chains, by)
    np.testing.assert_allclose(actual, _aggregate(oracle, responses, by))
    assert calls == [2, 2, 1, 2, 2, 1]


@pytest.mark.parametrize("by", ["person", "observation", "observed"])
def test_numpy_2pl_batches_bound_person_item_scratch(by, monkeypatch):
    model, responses, chains, oracle = _case("2PL")
    original = backend.sigmoid
    shapes = []

    def tracked(values):
        shapes.append(values.shape)
        return original(values)

    monkeypatch.setattr(backend, "sigmoid", tracked)
    monkeypatch.setattr(backend, "_entry_chunk_size", lambda n, p: 2)
    previous = mirt.get_backend()
    try:
        mirt.set_backend("numpy")
        actual = bayesian.compute_pointwise_log_lik(model, responses, chains, by)
    finally:
        mirt.set_backend(previous)
    np.testing.assert_allclose(actual, _aggregate(oracle, responses, by))
    assert shapes == [(2, 3), (2, 3), (1, 3)] * 2


@pytest.mark.parametrize("theta_layout", ["fixed", "sampled", "absent"])
def test_native_inputs_reuse_theta_storage(theta_layout):
    model, responses, chains, _ = _case("2PL")
    if theta_layout == "fixed":
        chains["theta"] = chains["theta"][0, :, None]
    elif theta_layout == "absent":
        del chains["theta"]
    _, _, theta = bayesian._batched_2pl_pointwise_inputs(
        model,
        chains,
        {k: v for k, v in chains.items() if k != "theta"},
        n_samples=2,
        n_persons=len(responses),
    )
    if theta_layout == "absent":
        assert theta.strides == (0, 0)
    else:
        assert np.shares_memory(theta, chains["theta"])
        if theta_layout == "fixed":
            assert theta.strides[0] == 0


@pytest.mark.parametrize("batch_size", [0, -1, True, np.bool_(True), 1.5, "2"])
def test_invalid_batch_sizes_fail_before_model_evaluation(batch_size):
    model, responses, chains, _ = _case("2PL")
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        bayesian.compute_pointwise_log_lik(
            model, responses, chains, batch_size=batch_size
        )


@pytest.mark.parametrize("bad", [np.uint64(2**63), float(2**64), 1j])
def test_response_codes_cannot_overflow_into_missing_values(bad):
    model = TwoParameterLogistic(1)
    with pytest.raises(ValueError, match="unsupported|integer category"):
        bayesian.compute_pointwise_log_lik(model, np.array([[bad]]), {})


def test_failed_custom_curve_restores_model_and_preserves_chains(monkeypatch):
    model, responses, chains, _ = _case("2PL")
    original = model.parameters
    original_chains = {k: v.copy() for k, v in chains.items()}

    def failing(theta, item_idx=None):
        model._parameters["difficulty"][:] = 100
        raise RuntimeError("curve failed")

    monkeypatch.setattr(model, "probability", failing)
    with pytest.raises(RuntimeError, match="curve failed"):
        bayesian.compute_pointwise_log_lik(model, responses, chains)
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)
    for name, values in original_chains.items():
        np.testing.assert_array_equal(chains[name], values)


@pytest.mark.parametrize("layout", ["broadcast", "negative_stride", "unaligned"])
def test_native_pointwise_accepts_strided_and_unaligned_chains(
    layout, selected_backend
):
    model, responses, chains, oracle = _case("2PL")
    if layout == "broadcast":
        chains = {
            name: np.broadcast_to(values[:1], values.shape)
            for name, values in chains.items()
        }
        oracle = np.broadcast_to(oracle[:1], oracle.shape)
    elif layout == "negative_stride":
        chains = {name: values[::-1] for name, values in chains.items()}
        oracle = oracle[::-1]
    else:
        for name, values in chains.items():
            storage = bytearray(values.nbytes + 1)
            unaligned = np.ndarray(
                values.shape, dtype=np.float64, buffer=storage, offset=1
            )
            unaligned[:] = values
            assert not unaligned.flags.aligned
            chains[name] = unaligned
    actual = bayesian.compute_pointwise_log_lik(model, responses, chains, by="observed")
    np.testing.assert_allclose(actual, oracle[:, responses >= 0])


@pytest.mark.parametrize("kind", ["2PL", "3PL", "GRM"])
@pytest.mark.parametrize("by", ["person", "observation", "observed"])
def test_empty_person_batches_keep_sample_dimension(kind, by, selected_backend):
    model, responses, chains, _ = _case(kind)
    chains["theta"] = chains["theta"][:, :0]
    actual = bayesian.compute_pointwise_log_lik(
        model, responses[:0], chains, by, batch_size=2
    )
    assert actual.shape == (2, 0)


@pytest.mark.parametrize("fixed_theta", [False, True])
@pytest.mark.parametrize("diagnostic", ["pointwise", "predictive"])
@pytest.mark.parametrize("fail", [False, True])
def test_custom_curves_can_mutate_theta_without_changing_chains(
    fixed_theta, diagnostic, fail, monkeypatch
):
    model, responses, chains, _ = _case("2PL")
    if fixed_theta:
        chains["theta"] = chains["theta"][0, :, None]
    original = {name: values.copy() for name, values in chains.items()}
    for values in chains.values():
        values.setflags(write=False)

    def curve(theta, item_idx=None):
        theta[:] += 10
        if fail:
            raise RuntimeError("mutating curve failed")
        return np.full((len(theta), model.n_items), 0.8)

    monkeypatch.setattr(model, "probability", curve)

    def run():
        if diagnostic == "pointwise":
            return bayesian.compute_pointwise_log_lik(
                model, responses, chains, batch_size=2
            )
        return bayesian.posterior_predictive_check(
            SimpleNamespace(chains=chains), responses, model, seed=324
        )

    if fail:
        with pytest.raises(RuntimeError, match="mutating curve failed"):
            run()
    else:
        run()
    for name, values in original.items():
        np.testing.assert_array_equal(chains[name], values)
