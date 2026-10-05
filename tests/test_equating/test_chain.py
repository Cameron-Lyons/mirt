"""Tests for chain linking."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.equating.chain import (
    ChainLinkingResult,
    TimePointModel,
    accumulate_constants,
    chain_link,
    chain_linking_summary,
    concurrent_link,
    detect_longitudinal_drift,
    transform_theta_to_reference,
    transform_to_reference,
)


@pytest.fixture
def linked_models():
    """Create multiple linked models for testing."""
    from mirt.models.dichotomous import TwoParameterLogistic

    rng = np.random.default_rng(42)
    n_items = 10

    disc = np.abs(rng.normal(1.0, 0.3, n_items))
    diff1 = rng.normal(0, 1, n_items)
    diff2 = diff1 + 0.5
    diff3 = diff2 + 0.5

    models = []
    for diff in [diff1, diff2, diff3]:
        model = TwoParameterLogistic(n_items=n_items)
        model._parameters = {
            "discrimination": disc.copy(),
            "difficulty": diff.copy(),
        }
        model._is_fitted = True
        model._n_factors = 1
        models.append(model)

    anchor_pairs = [
        (list(range(5)), list(range(5))),
        (list(range(5)), list(range(5))),
    ]

    return models, anchor_pairs


class TestChainLinkingResult:
    """Tests for ChainLinkingResult dataclass."""

    def test_initialization(self, linked_models):
        """Test ChainLinkingResult initialization."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs)

        assert isinstance(result, ChainLinkingResult)
        assert len(result.cumulative_A) == 3
        assert len(result.cumulative_B) == 3
        assert len(result.pairwise_results) == 2

    def test_reference_index(self, linked_models):
        """Test that reference index is stored."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs, reference_index=1)

        assert result.reference_index == 1


class TestChainLink:
    """Tests for chain_link function."""

    def test_basic_chain_link(self, linked_models):
        """Test basic chain linking."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs)

        assert len(result.cumulative_A) == len(models)
        assert len(result.cumulative_B) == len(models)

    def test_reference_identity(self, linked_models):
        """Test that reference model has identity transformation."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs, reference_index=0)

        assert result.cumulative_A[0] == pytest.approx(1.0)
        assert result.cumulative_B[0] == pytest.approx(0.0)

    def test_different_reference_indices(self, linked_models):
        """Test with different reference indices."""
        models, anchor_pairs = linked_models

        result0 = chain_link(models, anchor_pairs, reference_index=0)
        result1 = chain_link(models, anchor_pairs, reference_index=1)

        assert result0.cumulative_A[0] == pytest.approx(1.0)
        assert result1.cumulative_A[1] == pytest.approx(1.0)

    def test_invalid_anchor_pairs_length(self, linked_models):
        """Test that mismatched anchor pairs raise error."""
        models, _ = linked_models
        wrong_anchors = [(list(range(5)), list(range(5)))]

        with pytest.raises(ValueError, match="Expected .* anchor pairs"):
            chain_link(models, wrong_anchors)

    def test_invalid_reference_index(self, linked_models):
        """Test that invalid reference index raises error."""
        models, anchor_pairs = linked_models

        with pytest.raises(ValueError, match="Invalid reference_index"):
            chain_link(models, anchor_pairs, reference_index=10)

    def test_pairwise_results_computed(self, linked_models):
        """Test that pairwise results are computed."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs)

        assert len(result.pairwise_results) == 2
        for pr in result.pairwise_results:
            assert hasattr(pr, "constants")
            assert pr.constants.A is not None
            assert pr.constants.B is not None

    def test_drift_accumulation_computed(self, linked_models):
        """Test that drift accumulation is computed."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs, compute_drift=True)

        assert result.drift_accumulation is not None

    def test_no_drift_accumulation(self, linked_models):
        """Test without drift computation."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs, compute_drift=False)

        assert result.drift_accumulation is None

    def test_single_model_chain_is_an_identity(self, linked_models):
        models, _ = linked_models

        result = chain_link([models[0]], [], compute_drift=True)

        assert result.cumulative_A == [1.0]
        assert result.cumulative_B == [0.0]
        assert result.pairwise_results == []
        assert result.drift_accumulation is not None
        assert result.drift_accumulation.shape == (0, 0)

    def test_anchor_indices_are_validated(self, linked_models):
        models, anchor_pairs = linked_models
        anchor_pairs[0] = ([0, 1, 2], [0, 1, 20])

        with pytest.raises(ValueError, match="out-of-range right"):
            chain_link(models, anchor_pairs)


class TestAccumulateConstants:
    """Tests for accumulate_constants function."""

    def test_identity_at_reference(self):
        """Test that reference has identity transformation."""
        pairwise_A = [1.1, 0.9]
        pairwise_B = [0.2, -0.1]

        cum_A, cum_B = accumulate_constants(pairwise_A, pairwise_B, reference_index=0)

        assert cum_A[0] == pytest.approx(1.0)
        assert cum_B[0] == pytest.approx(0.0)

    def test_accumulation_forward(self):
        """Test forward accumulation."""
        pairwise_A = [1.0, 1.0]
        pairwise_B = [0.0, 0.0]

        cum_A, cum_B = accumulate_constants(pairwise_A, pairwise_B, reference_index=0)

        assert cum_A[0] == pytest.approx(1.0)
        assert cum_A[1] == pytest.approx(1.0)
        assert cum_A[2] == pytest.approx(1.0)

    def test_accumulation_with_shifts(self):
        """Test accumulation with non-identity transformations."""
        pairwise_A = [1.0, 1.0]
        pairwise_B = [0.5, 0.3]

        cum_A, cum_B = accumulate_constants(pairwise_A, pairwise_B, reference_index=0)

        assert cum_A[0] == pytest.approx(1.0)
        assert cum_B[0] == pytest.approx(0.0)
        assert cum_B[1] == pytest.approx(-0.5)
        assert cum_B[2] == pytest.approx(-0.8)

    def test_middle_reference(self):
        """Test with middle model as reference."""
        pairwise_A = [1.0, 1.0]
        pairwise_B = [0.5, 0.3]

        cum_A, cum_B = accumulate_constants(pairwise_A, pairwise_B, reference_index=1)

        assert cum_A[1] == pytest.approx(1.0)
        assert cum_B[1] == pytest.approx(0.0)

    @pytest.mark.parametrize("reference_index", [0, 1, 2, 3])
    def test_nonunit_affine_composition(self, reference_index):
        pairwise_A = [1.2, 0.75, 1.8]
        pairwise_B = [0.4, -0.3, 0.6]
        theta_by_time = [np.array([-1.5, 0.0, 2.0])]
        for slope, intercept in zip(pairwise_A, pairwise_B, strict=True):
            theta_by_time.append(slope * theta_by_time[-1] + intercept)

        cumulative_A, cumulative_B = accumulate_constants(
            pairwise_A,
            pairwise_B,
            reference_index=reference_index,
        )

        for time_index, theta in enumerate(theta_by_time):
            transformed = cumulative_A[time_index] * theta + cumulative_B[time_index]
            assert_allclose(transformed, theta_by_time[reference_index], atol=1e-14)

    @pytest.mark.parametrize(
        ("pairwise_A", "pairwise_B", "reference_index", "message"),
        [
            ([1.0], [], 0, "same length"),
            ([0.0], [0.0], 0, "positive"),
            ([np.nan], [0.0], 0, "positive"),
            ([1.0], [np.inf], 0, "finite"),
            ([1.0], [0.0], 2, "Invalid reference_index"),
        ],
    )
    def test_invalid_constants_are_rejected(
        self,
        pairwise_A,
        pairwise_B,
        reference_index,
        message,
    ):
        with pytest.raises(ValueError, match=message):
            accumulate_constants(pairwise_A, pairwise_B, reference_index)


class TestTransformToReference:
    """Tests for transform_to_reference function."""

    def test_basic_transformation(self, linked_models):
        """Test basic parameter transformation."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs, reference_index=0)

        transformed = transform_to_reference(
            models[2], chain_result, time_index=2, in_place=False
        )

        assert transformed is not models[2]
        assert transformed.is_fitted

    def test_reference_unchanged(self, linked_models):
        """Test that reference model transformation is identity."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs, reference_index=0)

        original_diff = np.array(models[0].difficulty).copy()

        transformed = transform_to_reference(
            models[0], chain_result, time_index=0, in_place=False
        )

        assert_allclose(transformed.difficulty, original_diff, atol=0.01)


class TestTransformThetaToReference:
    """Tests for transform_theta_to_reference function."""

    def test_basic_theta_transformation(self, linked_models):
        """Test basic theta transformation."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs, reference_index=0)
        theta = np.array([0.0, 1.0, -1.0])

        transformed = transform_theta_to_reference(theta, chain_result, time_index=2)

        assert transformed.shape == theta.shape

    def test_reference_theta_unchanged(self, linked_models):
        """Test that reference theta is unchanged."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs, reference_index=0)
        theta = np.array([0.0, 1.0, -1.0])

        transformed = transform_theta_to_reference(theta, chain_result, time_index=0)

        assert_allclose(transformed, theta)

    def test_invalid_time_index_is_rejected(self, linked_models):
        models, anchor_pairs = linked_models
        chain_result = chain_link(models, anchor_pairs)

        with pytest.raises(ValueError, match="Invalid time_index"):
            transform_theta_to_reference(np.array([0.0]), chain_result, time_index=-1)

    def test_nonfinite_theta_is_rejected(self, linked_models):
        models, anchor_pairs = linked_models
        chain_result = chain_link(models, anchor_pairs)

        with pytest.raises(ValueError, match="finite"):
            transform_theta_to_reference(
                np.array([0.0, np.nan]), chain_result, time_index=0
            )


class TestConcurrentLink:
    """Tests for concurrent_link function."""

    def test_basic_concurrent_link(self, linked_models):
        """Test basic concurrent linking."""
        models, _ = linked_models

        anchor_matrices = [
            [[(i, i) for i in range(5)]],
            [[(i, i) for i in range(5)]],
        ]

        result = concurrent_link(models, anchor_matrices)

        assert len(result) == len(models)
        for A, B in result:
            assert isinstance(A, float)
            assert isinstance(B, float)

    def test_reference_identity(self, linked_models):
        """Test that first model has identity transformation."""
        models, _ = linked_models

        anchor_matrices = [
            [[(i, i) for i in range(5)]],
            [[(i, i) for i in range(5)]],
        ]

        result = concurrent_link(models, anchor_matrices)

        assert result[0][0] == pytest.approx(1.0)
        assert result[0][1] == pytest.approx(0.0)

    def test_recovers_three_pl_transformation(self):
        from mirt.models.dichotomous import ThreeParameterLogistic

        forward_A, forward_B = 1.25, 0.35
        discrimination = np.array([0.7, 0.9, 1.1, 1.3, 1.5, 1.8])
        difficulty = np.linspace(-1.5, 1.5, 6)
        guessing = np.linspace(0.05, 0.25, 6)

        reference = ThreeParameterLogistic(6)
        reference.set_parameters(
            discrimination=discrimination,
            difficulty=difficulty,
            guessing=guessing,
        )
        transformed = ThreeParameterLogistic(6)
        transformed.set_parameters(
            discrimination=discrimination / forward_A,
            difficulty=forward_A * difficulty + forward_B,
            guessing=guessing,
        )
        pairs = [[[(index, index) for index in range(6)]]]

        result = concurrent_link(
            [reference, transformed],
            pairs,
            method="stocking_lord",
            max_iter=100,
            tol=1e-10,
        )

        assert result[1][0] == pytest.approx(1.0 / forward_A, rel=2e-5)
        assert result[1][1] == pytest.approx(-forward_B / forward_A, abs=2e-5)

    def test_supports_variable_category_polytomous_models(self):
        from mirt.models.polytomous import GradedResponseModel

        forward_A, forward_B = 1.25, 0.35
        categories = [3, 4, 3, 5]
        discrimination = np.array([0.8, 1.0, 1.2, 1.4])
        reference = GradedResponseModel(4, categories)
        thresholds = reference.thresholds.copy()
        reference.set_parameters(
            discrimination=discrimination,
            thresholds=thresholds,
        )
        transformed = GradedResponseModel(4, categories)
        transformed.set_parameters(
            discrimination=discrimination / forward_A,
            thresholds=forward_A * thresholds + forward_B,
        )
        pairs = [[[(index, index) for index in range(4)]]]

        result = concurrent_link(
            [reference, transformed],
            pairs,
            method="haebara",
            max_iter=100,
            tol=1e-10,
        )

        assert result[1][0] == pytest.approx(1.0 / forward_A, rel=2e-5)
        assert result[1][1] == pytest.approx(-forward_B / forward_A, abs=2e-5)

    def test_batches_probability_evaluations(self, linked_models, monkeypatch):
        from scipy import optimize

        models, _ = linked_models
        calls = [0, 0, 0]
        for model_index, model in enumerate(models):
            probability = model.probability

            def counted_probability(
                theta, item_idx=None, *, _index=model_index, _fn=probability
            ):
                calls[_index] += 1
                return _fn(theta, item_idx)

            monkeypatch.setattr(model, "probability", counted_probability)

        def evaluate_once(function, x0, **kwargs):
            return SimpleNamespace(x=x0, fun=function(x0), success=True)

        monkeypatch.setattr(optimize, "minimize", evaluate_once)
        anchor_matrices = [
            [[(index, index) for index in range(10)]],
            [[(index, index) for index in range(10)]],
        ]

        concurrent_link(models, anchor_matrices)

        assert calls == [1, 1, 1]

    def test_uses_itemwise_evaluation_for_sparse_anchor_banks(
        self, linked_models, monkeypatch
    ):
        from scipy import optimize

        models, _ = linked_models
        selected_calls: list[list[int | None]] = [[], []]
        for model_index, model in enumerate(models[:2]):
            probability = model.probability

            def counted_probability(
                theta, item_idx=None, *, _index=model_index, _fn=probability
            ):
                selected_calls[_index].append(item_idx)
                return _fn(theta, item_idx)

            monkeypatch.setattr(model, "probability", counted_probability)

        def evaluate_once(function, x0, **kwargs):
            return SimpleNamespace(x=x0, fun=function(x0), success=True)

        monkeypatch.setattr(optimize, "minimize", evaluate_once)

        concurrent_link(models[:2], [[[(2, 3), (7, 8)]]])

        assert selected_calls == [[2, 7], [3, 8]]

    def test_batches_dense_anchor_subsets(self, linked_models, monkeypatch):
        """Anchors covering a quarter of the bank use one batched evaluation."""
        from scipy import optimize

        models, _ = linked_models
        selected_calls: list[list[int | None]] = [[], []]
        for model_index, model in enumerate(models[:2]):
            probability = model.probability

            def counted_probability(
                theta, item_idx=None, *, _index=model_index, _fn=probability
            ):
                selected_calls[_index].append(item_idx)
                return _fn(theta, item_idx)

            monkeypatch.setattr(model, "probability", counted_probability)

        def evaluate_once(function, x0, **kwargs):
            return SimpleNamespace(x=x0, fun=function(x0), success=True)

        monkeypatch.setattr(optimize, "minimize", evaluate_once)

        concurrent_link(models[:2], [[[(2, 3), (7, 8), (4, 1)]]])

        assert selected_calls == [[None], [None]]

    def test_gradient_probes_reevaluate_only_the_perturbed_form(
        self, linked_models, monkeypatch
    ):
        """A coordinate probe touches one form's curves, not every free form."""
        from scipy import optimize

        models, _ = linked_models
        calls = [0, 0, 0]
        for model_index, model in enumerate(models):
            probability = model.probability

            def counted_probability(
                theta, item_idx=None, *, _index=model_index, _fn=probability
            ):
                calls[_index] += 1
                return _fn(theta, item_idx)

            monkeypatch.setattr(model, "probability", counted_probability)

        captured = {}

        def evaluate_value_and_gradient(function, x0, *, jac, **kwargs):
            x = np.array([0.1, -0.2, 0.3, 0.05])
            captured["fun"] = function
            captured["value"] = function(x)
            captured["gradient"] = jac(x)
            return SimpleNamespace(x=x0, fun=captured["value"], success=True)

        monkeypatch.setattr(optimize, "minimize", evaluate_value_and_gradient)
        anchor_matrices = [
            [[(index, index) for index in range(5)]],
            [[(index, index) for index in range(5)]],
        ]

        concurrent_link(models, anchor_matrices)

        # One value evaluation per free form plus two coordinate probes each.
        assert calls == [1, 3, 3]
        x = np.array([0.1, -0.2, 0.3, 0.05])
        expected = np.array(
            [
                (captured["fun"](x + 1e-8 * unit) - captured["value"])
                / ((x + 1e-8 * unit) - x)[position]
                for position, unit in enumerate(np.eye(4))
            ]
        )
        assert_allclose(captured["gradient"], expected, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("method", ["stocking_lord", "haebara"])
    @pytest.mark.parametrize("family", ["3PL", "GRM"])
    def test_matches_numerical_gradient_lbfgs_oracle(self, family, method):
        """The separable gradient reproduces whole-objective L-BFGS-B fits."""
        from scipy import optimize

        from mirt.models.dichotomous import ThreeParameterLogistic
        from mirt.models.polytomous import GradedResponseModel

        rng = np.random.default_rng(20)
        n_items, n_forms = 8, 4
        slopes = rng.uniform(0.7, 1.8, n_items)
        locations = np.sort(rng.normal(0.0, 1.0, (n_items, 4)), axis=1)
        guessing = rng.uniform(0.05, 0.2, n_items)
        models = []
        for scale, shift in zip(
            [1.0, 1.3, 0.8, 1.1], [0.0, 0.4, -0.3, 0.2], strict=True
        ):
            # Noisy affine copies of one calibration keep the fit well posed.
            form_slopes = slopes / scale * np.exp(rng.normal(0.0, 0.05, n_items))
            form_locations = scale * locations + shift
            form_locations += rng.normal(0.0, 0.05, (n_items, 1))
            if family == "3PL":
                model = ThreeParameterLogistic(n_items)
                model.set_parameters(
                    discrimination=form_slopes,
                    difficulty=form_locations[:, 0],
                    guessing=guessing,
                )
            else:
                model = GradedResponseModel(n_items, [3, 4, 3, 5] * 2)
                thresholds = model.thresholds.copy()
                for item, count in enumerate(model.n_categories):
                    thresholds[item, : count - 1] = form_locations[item, : count - 1]
                model.set_parameters(discrimination=form_slopes, thresholds=thresholds)
            models.append(model)
        assert len(models) == n_forms
        anchor_matrices = [
            [[(item, item) for item in range(5)], [(6, 6), (7, 7)]],
            [[(item, item) for item in range(5)]],
            [[(item, item) for item in range(5)]],
        ]
        relations = [(0, 1, range(5)), (0, 2, [6, 7]), (1, 2, range(5))]
        relations.append((2, 3, range(5)))
        theta = np.linspace(-4.0, 4.0, 61)
        weights = np.exp(-0.5 * theta**2)
        weights /= weights.sum()

        def item_curves(model, grid, item):
            values = np.asarray(model.probability(grid[:, None], item))
            if values.ndim == 1:
                return values, None
            return values @ np.arange(values.shape[1]), values

        def objective(parameters):
            scales = np.r_[1.0, np.exp(parameters[:3])]
            shifts = np.r_[0.0, parameters[3:]]
            loss = 0.0
            for left, right, items in relations:
                left_grid = (theta - shifts[left]) / scales[left]
                right_grid = (theta - shifts[right]) / scales[right]
                left_curves = [item_curves(models[left], left_grid, i) for i in items]
                right_curves = [
                    item_curves(models[right], right_grid, i) for i in items
                ]
                if method == "stocking_lord":
                    difference = sum(c[0] for c in left_curves) - sum(
                        c[0] for c in right_curves
                    )
                    loss += float(weights @ difference**2)
                    continue
                for (left_score, left_cat), (right_score, right_cat) in zip(
                    left_curves, right_curves, strict=True
                ):
                    if left_cat is None:
                        loss += float(weights @ (left_score - right_score) ** 2)
                    else:
                        loss += float(
                            np.sum(weights[:, None] * (left_cat - right_cat) ** 2)
                        )
            return loss

        max_log_slope = np.log(100.0)
        oracle = optimize.minimize(
            objective,
            np.zeros(6),
            method="L-BFGS-B",
            bounds=[(-max_log_slope, max_log_slope)] * 3 + [(None, None)] * 3,
            options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-10},
        )
        assert oracle.success

        result = concurrent_link(
            models, anchor_matrices, method=method, max_iter=200, tol=1e-10
        )

        fitted = np.r_[
            np.log([constants[0] for constants in result[1:]]),
            [constants[1] for constants in result[1:]],
        ]
        assert result[0] == (1.0, 0.0)
        assert objective(fitted) == pytest.approx(oracle.fun, rel=1e-10, abs=1e-15)
        # Batched and per-item curves differ in the last bit; the FD gradient
        # turns that into ~1e-8 drift along the flat optimum.
        assert_allclose(fitted, oracle.x, atol=1e-6)

    def test_stocking_lord_and_haebara_use_distinct_losses(self, monkeypatch):
        from scipy import optimize

        from mirt.models.dichotomous import TwoParameterLogistic

        reference = TwoParameterLogistic(2)
        reference.set_parameters(
            discrimination=np.ones(2),
            difficulty=np.array([-1.0, 1.0]),
        )
        reversed_items = TwoParameterLogistic(2)
        reversed_items.set_parameters(
            discrimination=np.ones(2),
            difficulty=np.array([1.0, -1.0]),
        )
        losses = []

        def capture_loss(function, x0, **kwargs):
            loss = function(x0)
            losses.append(loss)
            return SimpleNamespace(x=x0, fun=loss, success=True)

        monkeypatch.setattr(optimize, "minimize", capture_loss)
        pairs = [[[(0, 0), (1, 1)]]]

        concurrent_link([reference, reversed_items], pairs, method="stocking_lord")
        concurrent_link([reference, reversed_items], pairs, method="haebara")

        assert losses[0] == pytest.approx(0.0, abs=1e-15)
        assert losses[1] > 0.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"method": "unknown"}, "Unknown concurrent linking method"),
            ({"n_theta": 1}, "at least 2"),
            ({"max_iter": 0}, "positive"),
            ({"tol": 0.0}, "positive"),
            ({"reference_index": -1}, "Invalid reference_index"),
            ({"reference_index": 3}, "Invalid reference_index"),
            ({"reference_index": True}, "must be an integer"),
            ({"reference_index": 1.5}, "must be an integer"),
        ],
    )
    def test_invalid_configuration_is_rejected(self, linked_models, kwargs, message):
        models, _ = linked_models
        anchor_matrices = [
            [[(index, index) for index in range(5)]],
            [[(index, index) for index in range(5)]],
        ]

        with pytest.raises(ValueError, match=message):
            concurrent_link(models, anchor_matrices, **kwargs)

    def test_disconnected_anchor_design_is_rejected(self, linked_models):
        models, _ = linked_models
        anchor_matrices = [[[(index, index) for index in range(5)]]]

        with pytest.raises(ValueError, match="connect every model"):
            concurrent_link(models, anchor_matrices)


def _polytomous_chain(family):
    """Exact affine copies of one polytomous calibration on four metrics."""
    from mirt.equating.polytomous import transform_polytomous_parameters
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        GradedResponseModel,
        NominalResponseModel,
    )

    categories = [3, 4, 3, 5, 4, 3]
    slopes = np.array([0.7, 0.9, 1.1, 1.3, 1.5, 1.8])
    if family == "NRM":
        base = NominalResponseModel(6, categories)
        rng = np.random.default_rng(8)
        base.set_parameters(
            slopes=np.sort(rng.uniform(-1.5, 2.0, (6, 5)), axis=1),
            intercepts=rng.normal(0.0, 0.7, (6, 5)),
        )
        model_type = "nrm"
    else:
        model_type = "grm" if family == "GRM" else "gpcm"
        base = (GradedResponseModel if family == "GRM" else GeneralizedPartialCredit)(
            6, categories
        )
        locations = np.zeros((6, 4))
        for item, count in enumerate(categories):
            locations[item, : count - 1] = np.linspace(-1.2, 1.4, count - 1) + (
                0.15 * item - 0.4
            )
        key = "thresholds" if family == "GRM" else "steps"
        base.set_parameters(discrimination=slopes, **{key: locations})
    scales = np.array([1.0, 1.3, 0.72, 1.65])
    shifts = np.array([0.0, 0.4, -0.8, 0.25])
    models = [
        transform_polytomous_parameters(base, scale, shift, model_type=model_type)
        for scale, shift in zip(scales, shifts, strict=True)
    ]
    return base, models, scales, shifts


class TestPolytomousChainLink:
    """chain_link dispatches each adjacent pair to its family's linker."""

    @pytest.mark.parametrize("reference_index", [0, 2])
    @pytest.mark.parametrize("family", ["GRM", "GPCM", "NRM"])
    def test_recovers_calibrations_and_category_probabilities(
        self, family, reference_index
    ):
        base, models, scales, shifts = _polytomous_chain(family)
        pairs = [([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]), ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])]
        pairs.append(([0, 2, 3, 5], [0, 2, 3, 5]))

        result = chain_link(models, pairs, reference_index=reference_index)

        expected_scales = scales[reference_index] / scales
        expected_shifts = shifts[reference_index] - expected_scales * shifts
        assert_allclose(result.cumulative_A, expected_scales, atol=1e-7)
        assert_allclose(result.cumulative_B, expected_shifts, atol=1e-7)
        canonical_theta = np.array([-2.0, -0.5, 0.0, 1.1, 2.4])
        reference_theta = (
            scales[reference_index] * canonical_theta + shifts[reference_index]
        )
        expected_probabilities = base.probability(canonical_theta[:, None])
        for index, model in enumerate(models):
            linked = transform_to_reference(model, result, index)
            assert_allclose(
                linked.probability(reference_theta[:, None]),
                expected_probabilities,
                atol=1e-6,
            )
            assert linked is not model

    @pytest.mark.parametrize("family", ["GRM", "GPCM"])
    def test_ordered_drift_uses_reference_location_changes(self, family):
        _, models, _, _ = _polytomous_chain(family)
        pairs = [([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])] * 3

        result = chain_link(models, pairs, method="haebara")

        assert result.drift_accumulation.shape == (3, 5)
        assert np.all(np.isfinite(result.drift_difficulty_changes))
        assert_allclose(result.drift_difficulty_changes, 0.0, atol=1e-6)

    def test_nrm_drift_reports_z_scores_without_location_changes(self):
        """NRM intercept contrasts are logits, so no theta change is implied."""
        _, models, _, _ = _polytomous_chain("NRM")
        drifted = models[2].parameters["intercepts"].copy()
        drifted[0, 1:3] += np.array([1.5, -1.0])
        models[2].set_parameters(intercepts=drifted)
        pairs = [([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])] * 3

        result = chain_link(models, pairs)

        assert not np.any(np.isnan(result.drift_accumulation))
        assert np.all(np.isnan(result.drift_difficulty_changes))
        flagged = detect_longitudinal_drift(result)
        assert (0, 0) in flagged["flagged_item_ids"]
        assert set(flagged["drift_direction"]) == {"unknown"}

    def test_tcc_uses_polytomous_expected_score_matching(self):
        _, models, _, _ = _polytomous_chain("GRM")
        pairs = [([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])] * 3

        tcc = chain_link(models, pairs, method="tcc")
        stocking_lord = chain_link(models, pairs, method="stocking_lord")

        assert tcc.cumulative_A == stocking_lord.cumulative_A
        assert tcc.cumulative_B == stocking_lord.cumulative_B

    @pytest.mark.parametrize(
        ("family", "method"),
        [("GRM", "bisector"), ("GPCM", "orthogonal"), ("NRM", "mean_sigma")],
    )
    def test_rejects_methods_the_family_cannot_use(self, family, method):
        _, models, _, _ = _polytomous_chain(family)
        pairs = [([0, 1, 2], [0, 1, 2])] * 3

        with pytest.raises(ValueError, match="not available"):
            chain_link(models, pairs, method=method)

    def test_rejects_mixed_response_families(self, linked_models):
        dichotomous, _ = linked_models
        _, polytomous, _, _ = _polytomous_chain("GRM")

        with pytest.raises(ValueError, match="different response families"):
            chain_link([dichotomous[0], polytomous[1]], [([0, 1, 2], [0, 1, 2])])


class TestDetectLongitudinalDrift:
    """Tests for detect_longitudinal_drift function."""

    def test_basic_drift_detection(self, linked_models):
        """Test basic drift detection."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs, compute_drift=True)

        drift_result = detect_longitudinal_drift(chain_result)

        assert "consistently_flagged" in drift_result
        assert "drift_direction" in drift_result

    def test_drift_detection_no_accumulation(self, linked_models):
        """Test drift detection without drift accumulation."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs, compute_drift=False)

        drift_result = detect_longitudinal_drift(chain_result)

        assert drift_result["consistently_flagged"] == []
        assert drift_result["drift_direction"] == []


class TestChainLinkingSummary:
    """Tests for chain_linking_summary function."""

    def test_basic_summary(self, linked_models):
        """Test basic summary generation."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs)

        summary = chain_linking_summary(chain_result)

        assert isinstance(summary, str)
        assert "Chain Linking" in summary
        assert "Cumulative" in summary

    def test_summary_contains_all_time_points(self, linked_models):
        """Test that summary contains all time points."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs)

        summary = chain_linking_summary(chain_result)

        for t in range(len(models)):
            assert str(t) in summary


class TestTimePointModel:
    """Tests for TimePointModel dataclass."""

    def test_initialization(self, linked_models):
        """Test TimePointModel initialization."""
        models, _ = linked_models

        tp = TimePointModel(
            model=models[0],
            anchor_items=[0, 1, 2],
            time_label="T1",
        )

        assert tp.model is models[0]
        assert tp.anchor_items == [0, 1, 2]
        assert tp.time_label == "T1"

    def test_default_time_label(self, linked_models):
        """Test default time label."""
        models, _ = linked_models

        tp = TimePointModel(
            model=models[0],
            anchor_items=[0, 1, 2],
        )

        assert tp.time_label == ""
