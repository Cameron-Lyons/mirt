"""Tests for local dependence statistics."""

import time
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import mirt.diagnostics.residuals as residuals_module
from mirt.diagnostics.ld import (
    LDResult,
    _compute_ld_chi2_g2,
    _compute_q3,
    compute_ld_chi2,
    compute_ld_statistics,
    compute_q3,
    flag_ld_pairs,
    ld_summary_table,
)


class _CountingPolytomousModel:
    """Small deterministic model that records probability evaluations."""

    is_polytomous = True

    def __init__(self, n_items: int) -> None:
        self.n_items = n_items
        self.n_categories = [2] * n_items
        self.max_categories = 2
        self.item_names = [f"Item_{index}" for index in range(n_items)]
        self.probability_calls = 0

    def positive_probability(self, theta, item_idx):
        theta_values = np.asarray(theta, dtype=np.float64).reshape(-1)
        slope = 0.8 + 0.1 * item_idx
        difficulty = -0.6 + 0.25 * item_idx
        logits = slope * (theta_values - difficulty)
        return 1.0 / (1.0 + np.exp(-logits))

    def probability(self, theta, item_idx=None):
        self.probability_calls += 1
        if item_idx is None:
            return np.stack(
                [
                    np.column_stack(
                        (
                            1.0 - self.positive_probability(theta, index),
                            self.positive_probability(theta, index),
                        )
                    )
                    for index in range(self.n_items)
                ],
                axis=1,
            )
        positive = self.positive_probability(theta, item_idx)
        return np.column_stack((1.0 - positive, positive))


def _scalar_gamma(table):
    """Goodman-Kruskal gamma by explicit concordant and discordant pairs."""
    concordant = discordant = 0.0
    rows, columns = table.shape
    for k in range(rows):
        for m in range(columns):
            for other_k in range(k + 1, rows):
                for other_m in range(columns):
                    if other_m > m:
                        concordant += table[k, m] * table[other_k, other_m]
                    elif other_m < m:
                        discordant += table[k, m] * table[other_k, other_m]
    return (concordant - discordant) / (concordant + discordant)


def _reference_marginal_ld(model, responses, n_quadpts=21, n_categories=None):
    """Scalar Chen-Thissen reference: integrate each expected cell node by node."""
    n_items = responses.shape[1]
    if n_categories is None:
        n_categories = getattr(model, "n_categories", [2] * n_items)
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_quadpts)
    weights = weights / weights.sum()
    curves = []
    for item in range(n_items):
        values = np.asarray(model.probability(nodes[:, None], item), dtype=float)
        if values.ndim == 1:
            values = np.column_stack((1.0 - values, values))
        curves.append(values[:, : n_categories[item]])

    chi2 = np.full((n_items, n_items), np.nan)
    g2 = np.full_like(chi2, np.nan)
    cramers_v = np.full_like(chi2, np.nan)
    for first in range(n_items):
        for second in range(first + 1, n_items):
            valid = (responses[:, first] >= 0) & (responses[:, second] >= 0)
            n_joint = int(valid.sum())
            if n_joint < 10:
                continue
            observed = np.zeros((n_categories[first], n_categories[second]))
            for left, right in zip(
                responses[valid, first], responses[valid, second], strict=True
            ):
                observed[int(left), int(right)] += 1
            expected = np.zeros_like(observed)
            for node in range(n_quadpts):
                expected += weights[node] * np.outer(
                    curves[first][node], curves[second][node]
                )
            expected *= n_joint

            statistic = np.sum((observed - expected) ** 2 / expected)
            positive = observed > 0
            likelihood = 2.0 * np.sum(
                observed[positive] * np.log(observed[positive] / expected[positive])
            )
            sign = np.sign(_scalar_gamma(observed) - _scalar_gamma(expected)) or 1.0
            smaller = min(n_categories[first], n_categories[second]) - 1
            chi2[first, second] = chi2[second, first] = statistic
            g2[first, second] = g2[second, first] = likelihood
            cramers_v[first, second] = cramers_v[second, first] = sign * np.sqrt(
                statistic / (n_joint * smaller)
            )
    return chi2, g2, cramers_v


# R mirt 1.48 reference: simdata() samples with set.seed(20261011), fitted by
# mirt(), then residuals(type = "LD"), residuals(type = "LDG2") and
# residuals(type = "LD", df.p = TRUE). Patterns are "category codes:count",
# with "." for a missing response.
_R_BINARY_PATTERNS = (
    "..00:1 ..10:1 ..11:1 .00.:1 .000:5 .001:1 .010:9 .011:4 .1.0:1 .1.1:1 "
    ".100:2 .11.:1 .110:6 .111:9 0..0:1 0.0.:1 0.00:3 0.1.:2 0.10:4 0.11:1 "
    "00.0:3 00.1:3 000.:2 0000:46 0001:6 001.:6 0010:28 0011:4 01.0:3 010.:1 "
    "0100:6 0101:3 011.:2 0110:20 0111:11 1...:1 1..0:2 1.00:1 1.1.:3 1.10:7 "
    "1.11:12 10..:1 10.0:8 10.1:5 100.:2 1000:23 1001:10 101.:7 1010:45 "
    "1011:30 11.0:5 11.1:7 1100:6 1101:3 111.:8 1110:54 1111:61"
)
_R_BINARY_SLOPES = [
    1.2959663311956207,
    1.3907653663364969,
    2.5657218477742489,
    1.0522550035220606,
]
_R_BINARY_INTERCEPTS = [
    0.87470757704440893,
    -0.21493353384505015,
    1.94599057498008254,
    -0.64099250744826897,
]
# Upper-triangle pairs (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3).
_R_BINARY_X2 = [
    1.14729407215572832,
    0.16969774100716611,
    1.99500932377661511,
    0.775384022100485182,
    0.039068068054272424,
    0.717810379860092884,
]
_R_BINARY_V = [
    -0.052327545422255435,
    0.020148832644968365,
    0.0688385029355320921,
    0.042864937839366027,
    -0.0095540846153404514,
    -0.0411940681727883395,
]
_R_BINARY_G2 = [
    1.1353852854942792,
    0.1688720492692265,
    2.0781725790890651,
    0.768901063099351467,
    0.038949505465528667,
    0.691526022385308137,
]
_R_BINARY_P = [
    0.28411631225268674,
    0.68038057459022161,
    0.15781809459308810,
    0.37855659820548571,
    0.84331387351889686,
    0.39686308478598870,
]
_R_GRADED_PATTERNS = (
    "000:48 001:41 002:14 010:12 011:12 012:7 020:19 021:25 022:19 100:20 "
    "101:32 102:19 110:7 111:19 112:6 120:11 121:33 122:18 200:18 201:26 "
    "202:25 210:6 211:17 212:19 220:12 221:46 222:69"
)
_R_GRADED_SLOPES = [1.27412134514996067, 0.85698125443298923, 1.34463859439424116]
_R_GRADED_INTERCEPTS = [
    [0.94087095298441903, -0.54465219070715287],
    [0.44386645651343271, -0.37576038183940441],
    [1.42213572210762251, -0.96976388142451464],
]
_R_GRADED_X2 = [1.1807385984176215, 3.1753245980415690, 0.611143732359126113]
_R_GRADED_V = [-0.031367958684216046, -0.051440293853178054, -0.022567375647881725]
_R_GRADED_G2 = [1.1863101398091789, 3.1588930024593576, 0.611769472857663921]
_R_GRADED_P = [0.88125958678605487, 0.52892564316029322, 0.96181735139288171]
# residuals(type = "Q3", method = "EAP") of the graded fit.
_R_GRADED_Q3 = [-0.18874935894258285, -0.33138558849903288, -0.19910562273389060]


def _expand_patterns(encoded):
    """Expand "codes:count" pattern counts into a response matrix."""
    rows = []
    for entry in encoded.split():
        codes, count = entry.split(":")
        pattern = [-1 if code == "." else int(code) for code in codes]
        rows.extend([pattern] * int(count))
    return np.array(rows)


def _r_binary_case():
    from mirt.models.dichotomous import TwoParameterLogistic

    slopes = np.array(_R_BINARY_SLOPES)
    model = TwoParameterLogistic(4)
    model.set_parameters(
        discrimination=slopes, difficulty=-np.array(_R_BINARY_INTERCEPTS) / slopes
    )
    model._is_fitted = True
    return model, _expand_patterns(_R_BINARY_PATTERNS)


def _r_graded_case():
    from mirt.models.polytomous import GradedResponseModel

    slopes = np.array(_R_GRADED_SLOPES)
    model = GradedResponseModel(3, n_categories=3)
    model.set_parameters(
        discrimination=slopes,
        thresholds=-np.array(_R_GRADED_INTERCEPTS) / slopes[:, None],
    )
    model._is_fitted = True
    return model, _expand_patterns(_R_GRADED_PATTERNS)


class TestLDResult:
    """Tests for LDResult dataclass."""

    @pytest.fixture
    def sample_ld_result(self):
        """Create sample LDResult for testing."""
        n_items = 5
        q3 = np.random.randn(n_items, n_items) * 0.1
        q3 = (q3 + q3.T) / 2
        np.fill_diagonal(q3, 1.0)

        chi2 = np.abs(np.random.randn(n_items, n_items) * 2)
        chi2 = (chi2 + chi2.T) / 2
        np.fill_diagonal(chi2, np.nan)

        return LDResult(
            q3_matrix=q3,
            ld_chi2_matrix=chi2,
            g2_matrix=chi2.copy(),
            adj_residual_corr=q3 + 1 / (n_items - 1),
            q3_flagged=[(0, 1, 0.25)],
            chi2_flagged=[(0, 2, 5.0, 0.02)],
            item_names=None,
        )

    def test_initialization(self, sample_ld_result):
        """Test LDResult initialization."""
        assert sample_ld_result.q3_matrix.shape == (5, 5)
        assert sample_ld_result.ld_chi2_matrix.shape == (5, 5)
        assert len(sample_ld_result.q3_flagged) == 1
        assert len(sample_ld_result.chi2_flagged) == 1

    def test_summary(self, sample_ld_result):
        """Test summary method."""
        summary = sample_ld_result.summary()

        assert "Local Dependence" in summary
        assert "Q3" in summary


class TestComputeLDStatistics:
    """Tests for compute_ld_statistics function."""

    def test_basic_ld_statistics(self, fitted_2pl_model, dichotomous_responses):
        """Test basic LD statistics computation."""
        model = fitted_2pl_model.model
        result = compute_ld_statistics(model, dichotomous_responses["responses"])

        assert isinstance(result, LDResult)
        n_items = dichotomous_responses["n_items"]
        assert result.q3_matrix.shape == (n_items, n_items)
        assert result.ld_chi2_matrix.shape == (n_items, n_items)

    def test_q3_matrix_symmetric(self, fitted_2pl_model, dichotomous_responses):
        """Test that Q3 matrix is symmetric."""
        model = fitted_2pl_model.model
        result = compute_ld_statistics(model, dichotomous_responses["responses"])

        assert_allclose(result.q3_matrix, result.q3_matrix.T)

    def test_chi2_matrix_symmetric(self, fitted_2pl_model, dichotomous_responses):
        """Test that chi-square matrix is symmetric."""
        model = fitted_2pl_model.model
        result = compute_ld_statistics(model, dichotomous_responses["responses"])

        valid_mask = ~np.isnan(result.ld_chi2_matrix)
        upper_tri = result.ld_chi2_matrix[valid_mask]
        lower_tri = result.ld_chi2_matrix.T[valid_mask]
        assert_allclose(upper_tri, lower_tri)

    def test_ld_with_provided_theta(self, fitted_2pl_model, dichotomous_responses):
        """Test LD statistics with provided theta."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_ld_statistics(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert isinstance(result, LDResult)

    def test_ld_flagged_pairs(self, fitted_2pl_model, dichotomous_responses):
        """Test that flagged pairs are returned."""
        model = fitted_2pl_model.model
        result = compute_ld_statistics(
            model,
            dichotomous_responses["responses"],
            q3_threshold=0.05,
        )

        for i, j, q3_val in result.q3_flagged:
            assert abs(q3_val) > 0.05
            assert i < j


class TestVectorizedLDCalculations:
    """Regression tests for the shared pairwise matrix calculations."""

    def test_q3_matches_pairwise_reference_with_missing_responses(self):
        rng = np.random.default_rng(90210)
        responses = rng.integers(0, 2, size=(40, 7))
        responses[rng.random(responses.shape) < 0.2] = -1
        residuals = rng.normal(size=responses.shape)
        residuals[responses < 0] = np.nan

        expected = np.zeros((responses.shape[1], responses.shape[1]))
        for first in range(responses.shape[1]):
            for second in range(first + 1, responses.shape[1]):
                valid = (responses[:, first] >= 0) & (responses[:, second] >= 0)
                valid &= ~np.isnan(residuals[:, first])
                valid &= ~np.isnan(residuals[:, second])
                if valid.sum() > 2:
                    correlation = np.corrcoef(
                        residuals[valid, first], residuals[valid, second]
                    )[0, 1]
                    expected[first, second] = expected[second, first] = correlation

        actual = _compute_q3(residuals, responses)

        assert_allclose(actual, expected, rtol=1e-13, atol=1e-14, equal_nan=True)

    def test_chi2_and_g2_match_pairwise_reference(self):
        rng = np.random.default_rng(314159)
        n_persons, n_items = 50, 6
        theta = rng.normal(size=(n_persons, 1))
        model = _CountingPolytomousModel(n_items)
        probabilities = np.column_stack(
            [model.positive_probability(theta, item) for item in range(n_items)]
        )
        responses = (rng.random(probabilities.shape) < probabilities).astype(int)
        responses[rng.random(responses.shape) < 0.15] = -1

        expected_chi2, expected_g2, expected_v = _reference_marginal_ld(
            model, responses
        )
        model.probability_calls = 0
        statistics = _compute_ld_chi2_g2(model, responses, n_quadpts=21)

        # One all-item evaluation on the quadrature grid.
        assert model.probability_calls == 1
        assert_allclose(
            statistics.chi2, expected_chi2, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        assert_allclose(
            statistics.g2, expected_g2, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        assert_allclose(
            statistics.cramers_v, expected_v, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_polytomous_tables_use_every_category(self):
        from mirt.models.polytomous import GradedResponseModel

        rng = np.random.default_rng(2718)
        model = GradedResponseModel(5, n_categories=[3, 4, 2, 5, 3])
        theta = rng.normal(size=(400, 1))
        probabilities = model.probability(theta)
        cumulative = np.cumsum(probabilities, axis=2)
        draws = rng.random((400, 5, 1))
        responses = np.minimum(
            (draws > cumulative).sum(axis=2), np.array(model.n_categories) - 1
        )
        responses[rng.random(responses.shape) < 0.1] = -1

        expected_chi2, expected_g2, expected_v = _reference_marginal_ld(
            model, responses, n_quadpts=15
        )
        result = compute_ld_statistics(model, responses, theta=theta, n_quadpts=15)

        assert_allclose(result.ld_chi2_matrix, expected_chi2, rtol=1e-11)
        assert_allclose(result.g2_matrix, expected_g2, rtol=1e-11, atol=1e-11)
        assert_allclose(result.cramers_v_matrix, expected_v, rtol=1e-11)
        categories = np.array(model.n_categories)
        degrees = np.multiply.outer(categories - 1, categories - 1).astype(float)
        np.fill_diagonal(degrees, np.nan)
        assert_allclose(result.chi2_df_matrix, degrees, equal_nan=True)
        rows, columns = np.triu_indices(5, k=1)
        assert_allclose(
            result.chi2_p_value_matrix[rows, columns],
            stats.chi2.sf(expected_chi2[rows, columns], degrees[rows, columns]),
            rtol=1e-10,
        )

    def test_full_analysis_reuses_one_bounded_probability_pass(self, monkeypatch):
        rng = np.random.default_rng(271828)
        n_persons, n_items = 40, 5
        theta = rng.normal(size=(n_persons, 1))
        model = _CountingPolytomousModel(n_items)
        probabilities = np.column_stack(
            [model.positive_probability(theta, item) for item in range(n_items)]
        )
        responses = (rng.random(probabilities.shape) < probabilities).astype(int)
        responses[rng.random(responses.shape) < 0.1] = -1

        monkeypatch.setattr(
            residuals_module,
            "_RESIDUAL_MAX_PROBABILITY_VALUES",
            0,
        )
        expected = compute_ld_statistics(model, responses, theta=theta)
        # Itemwise residuals plus one evaluation on the quadrature grid.
        assert model.probability_calls == n_items + 1

        model.probability_calls = 0
        monkeypatch.setattr(
            residuals_module,
            "_RESIDUAL_MAX_PROBABILITY_VALUES",
            responses.size * model.max_categories,
        )
        result = compute_ld_statistics(model, responses, theta=theta)

        assert model.probability_calls == 2
        assert_allclose(result.q3_matrix, expected.q3_matrix, equal_nan=True)
        assert_allclose(
            result.ld_chi2_matrix,
            expected.ld_chi2_matrix,
            equal_nan=True,
        )
        assert_allclose(result.g2_matrix, expected.g2_matrix, equal_nan=True)
        assert_allclose(
            result.adj_residual_corr,
            result.q3_matrix + 1.0 / (n_items - 1),
            equal_nan=True,
        )

    def test_single_item_adjustment_remains_finite(self):
        model = _CountingPolytomousModel(1)
        theta = np.linspace(-2.0, 2.0, 12).reshape(-1, 1)
        responses = np.array([[0], [1]] * 6)

        result = compute_ld_statistics(model, responses, theta=theta)

        assert_allclose(result.q3_matrix, np.zeros((1, 1)))
        assert_allclose(result.adj_residual_corr, np.zeros((1, 1)))


class TestComputeQ3:
    """Tests for compute_q3 function."""

    def test_basic_q3(self, fitted_2pl_model, dichotomous_responses):
        """Test basic Q3 computation."""
        model = fitted_2pl_model.model
        q3_matrix = compute_q3(model, dichotomous_responses["responses"])

        n_items = dichotomous_responses["n_items"]
        assert q3_matrix.shape == (n_items, n_items)

    def test_q3_diagonal_is_zero(self, fitted_2pl_model, dichotomous_responses):
        """Test that Q3 diagonal is zero (self-correlation excluded)."""
        model = fitted_2pl_model.model
        q3_matrix = compute_q3(model, dichotomous_responses["responses"])

        diagonal = np.diag(q3_matrix)
        assert_allclose(diagonal, np.zeros_like(diagonal))

    def test_q3_symmetric(self, fitted_2pl_model, dichotomous_responses):
        """Test that Q3 is symmetric."""
        model = fitted_2pl_model.model
        q3_matrix = compute_q3(model, dichotomous_responses["responses"])

        assert_allclose(q3_matrix, q3_matrix.T)

    def test_q3_values_bounded(self, fitted_2pl_model, dichotomous_responses):
        """Test that Q3 values are bounded by -1 and 1."""
        model = fitted_2pl_model.model
        q3_matrix = compute_q3(model, dichotomous_responses["responses"])

        non_diagonal = q3_matrix[~np.eye(q3_matrix.shape[0], dtype=bool)]
        assert np.all(non_diagonal >= -1.0)
        assert np.all(non_diagonal <= 1.0)

    def test_q3_with_theta(self, fitted_2pl_model, dichotomous_responses):
        """Test Q3 with provided theta."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        q3_matrix = compute_q3(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert q3_matrix.shape[0] == dichotomous_responses["n_items"]


class TestComputeLDChi2:
    """Tests for compute_ld_chi2 function."""

    def test_basic_ld_chi2(self, fitted_2pl_model, dichotomous_responses):
        """Test basic LD chi-square computation."""
        model = fitted_2pl_model.model
        chi2_matrix, p_matrix = compute_ld_chi2(
            model, dichotomous_responses["responses"]
        )

        n_items = dichotomous_responses["n_items"]
        assert chi2_matrix.shape == (n_items, n_items)
        assert p_matrix.shape == (n_items, n_items)

    def test_chi2_non_negative(self, fitted_2pl_model, dichotomous_responses):
        """Test that chi-square values are non-negative."""
        model = fitted_2pl_model.model
        chi2_matrix, _ = compute_ld_chi2(model, dichotomous_responses["responses"])

        valid_chi2 = chi2_matrix[~np.isnan(chi2_matrix)]
        assert np.all(valid_chi2 >= 0)

    def test_p_values_in_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that p-values are in [0, 1]."""
        model = fitted_2pl_model.model
        _, p_matrix = compute_ld_chi2(model, dichotomous_responses["responses"])

        valid_p = p_matrix[~np.isnan(p_matrix)]
        assert np.all(valid_p >= 0)
        assert np.all(valid_p <= 1)


class TestFlagLDPairs:
    """Tests for flag_ld_pairs function."""

    def test_flag_q3(self, fitted_2pl_model, dichotomous_responses):
        """Test flagging by Q3."""
        model = fitted_2pl_model.model
        ld_result = compute_ld_statistics(model, dichotomous_responses["responses"])

        flagged = flag_ld_pairs(ld_result, q3_threshold=0.05, method="q3")

        assert isinstance(flagged, list)
        for pair in flagged:
            assert len(pair) == 2
            i, j = pair
            assert abs(ld_result.q3_matrix[i, j]) > 0.05

    def test_flag_chi2(self, fitted_2pl_model, dichotomous_responses):
        """Test flagging by chi-square."""
        model = fitted_2pl_model.model
        ld_result = compute_ld_statistics(model, dichotomous_responses["responses"])

        flagged = flag_ld_pairs(ld_result, chi2_alpha=0.05, method="chi2")

        assert isinstance(flagged, list)

    def test_flag_both(self, fitted_2pl_model, dichotomous_responses):
        """Test flagging by both Q3 and chi-square."""
        model = fitted_2pl_model.model
        ld_result = compute_ld_statistics(model, dichotomous_responses["responses"])

        flagged = flag_ld_pairs(ld_result, method="both")

        assert isinstance(flagged, list)

    def test_stricter_threshold_fewer_flags(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that stricter threshold produces fewer flags."""
        model = fitted_2pl_model.model
        ld_result = compute_ld_statistics(model, dichotomous_responses["responses"])

        lenient = flag_ld_pairs(ld_result, q3_threshold=0.05, method="q3")
        strict = flag_ld_pairs(ld_result, q3_threshold=0.3, method="q3")

        assert len(strict) <= len(lenient)

    def test_q3_threshold_uses_full_matrix(self):
        """A lower threshold can identify pairs absent from cached flags."""
        q3 = np.array(
            [
                [0.0, 0.10, -0.03],
                [0.10, 0.0, -0.08],
                [-0.03, -0.08, 0.0],
            ]
        )
        chi2 = np.full((3, 3), np.nan)
        result = LDResult(q3, chi2, chi2.copy(), q3.copy(), [], [])

        assert flag_ld_pairs(result, q3_threshold=0.05, method="q3") == [
            (0, 1),
            (1, 2),
        ]

    def test_chi2_matches_survival_function_with_nonfinite_values(self):
        """Matrix selection preserves the scalar chi-square definition."""
        q3 = np.zeros((4, 4))
        chi2 = np.array(
            [
                [np.nan, 0.1, 4.0, np.inf],
                [0.1, np.nan, 6.0, 2.0],
                [4.0, 6.0, np.nan, 8.0],
                [np.inf, 2.0, 8.0, np.nan],
            ]
        )
        result = LDResult(q3, chi2, chi2.copy(), q3.copy(), [], [])

        assert flag_ld_pairs(result, method="chi2") == [
            (0, 2),
            (0, 3),
            (1, 2),
            (2, 3),
        ]

    @pytest.mark.parametrize("method", ["", "Q3", "unknown"])
    def test_rejects_unknown_method(self, method):
        """Pair flagging rejects unsupported selection methods."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(ValueError, match="method"):
            flag_ld_pairs(result, method=method)

    @pytest.mark.parametrize("method", [None, 1, True])
    def test_rejects_non_string_method(self, method):
        """Pair flagging requires a string selection method."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(TypeError, match="method"):
            flag_ld_pairs(result, method=method)

    @pytest.mark.parametrize("threshold", [-0.1, np.inf, np.nan])
    def test_rejects_invalid_q3_threshold(self, threshold):
        """Q3 thresholds must be finite and nonnegative."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(ValueError, match="q3_threshold"):
            flag_ld_pairs(result, q3_threshold=threshold, method="q3")

    @pytest.mark.parametrize("threshold", [None, "0.2", True])
    def test_rejects_non_numeric_q3_threshold(self, threshold):
        """Q3 thresholds reject non-numeric and boolean values."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(TypeError, match="q3_threshold"):
            flag_ld_pairs(result, q3_threshold=threshold, method="q3")

    @pytest.mark.parametrize("alpha", [0.0, 1.0, np.inf, np.nan])
    def test_rejects_invalid_chi2_alpha(self, alpha):
        """Chi-square levels must be finite probabilities."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(ValueError, match="chi2_alpha"):
            flag_ld_pairs(result, chi2_alpha=alpha, method="chi2")


class TestLDSummaryTable:
    """Tests for ld_summary_table function."""

    def test_summary_table(self, fitted_2pl_model, dichotomous_responses):
        """Test summary table generation."""
        model = fitted_2pl_model.model
        ld_result = compute_ld_statistics(model, dichotomous_responses["responses"])

        table = ld_summary_table(ld_result)

        assert isinstance(table, str)
        assert "Q3" in table
        assert "Item" in table

    def test_summary_table_top_n(self, fitted_2pl_model, dichotomous_responses):
        """Test summary table with top_n parameter."""
        model = fitted_2pl_model.model
        ld_result = compute_ld_statistics(model, dichotomous_responses["responses"])

        table = ld_summary_table(ld_result, top_n=5)

        lines = [
            line
            for line in table.split("\n")
            if line.strip() and not line.startswith("-")
        ]
        assert len(lines) <= 7

    def test_summary_table_stably_orders_ties_and_missing_q3(self):
        """Finite top pairs precede missing values and ties keep matrix order."""
        q3 = np.array(
            [
                [0.0, 0.5, -0.5, np.nan],
                [0.5, 0.0, 0.2, 0.1],
                [-0.5, 0.2, 0.0, 0.3],
                [np.nan, 0.1, 0.3, 0.0],
            ]
        )
        chi2 = np.full((4, 4), np.nan)
        result = LDResult(
            q3,
            chi2,
            chi2.copy(),
            q3.copy(),
            [],
            [],
            ["one", "two", "three", "four"],
        )

        lines = ld_summary_table(result, top_n=3).splitlines()[2:]

        assert [line.split()[:2] for line in lines] == [
            ["one", "two"],
            ["one", "three"],
            ["three", "four"],
        ]

    def test_summary_table_zero_rows(self):
        """A zero limit returns only the table heading."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        assert len(ld_summary_table(result, top_n=0).splitlines()) == 2

    @pytest.mark.parametrize("top_n", [-1, -10])
    def test_summary_table_rejects_negative_limit(self, top_n):
        """Negative display limits are rejected instead of slicing implicitly."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(ValueError, match="top_n"):
            ld_summary_table(result, top_n=top_n)

    @pytest.mark.parametrize("top_n", [None, 1.5, True])
    def test_summary_table_rejects_non_integer_limit(self, top_n):
        """Display limits reject non-integer and boolean values."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(TypeError, match="top_n"):
            ld_summary_table(result, top_n=top_n)


class TestLDWithLocallyDependentData:
    """Tests with data having known local dependence."""

    def test_detects_correlated_items(self):
        """Test that LD is detected in correlated items."""
        rng = np.random.default_rng(42)
        n_persons = 200
        n_items = 8

        theta = rng.standard_normal(n_persons)
        diff = rng.normal(0, 1, n_items)

        probs = 1 / (1 + np.exp(-(theta[:, None] - diff)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        correlation = rng.random((n_persons,)) < 0.7
        responses[correlation, 1] = responses[correlation, 0]

        from mirt import fit_mirt

        result = fit_mirt(responses, model="2PL", max_iter=20, n_quadpts=11)

        ld_result = compute_ld_statistics(result.model, responses)

        q3_01 = abs(ld_result.q3_matrix[0, 1])
        q3_others = np.abs(ld_result.q3_matrix)
        np.fill_diagonal(q3_others, 0)
        q3_others[0, 1] = 0
        q3_others[1, 0] = 0

        assert q3_01 > np.mean(q3_others)


def _simulated_binary_model(model_name, *, n_persons=600, n_items=8, seed=0):
    """Return a parameterized dichotomous model, simulated responses and theta."""
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        OneParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )

    rng = np.random.default_rng(seed)
    n_factors = 2 if model_name == "2PL-2D" else 1
    if model_name == "1PL":
        model = OneParameterLogistic(n_items=n_items)
        model.set_parameters(difficulty=rng.normal(size=n_items))
    elif model_name in {"2PL", "2PL-2D"}:
        model = TwoParameterLogistic(n_items=n_items, n_factors=n_factors)
        slopes = rng.uniform(0.8, 2.0, size=(n_items, n_factors))
        model.set_parameters(
            discrimination=slopes if n_factors > 1 else slopes[:, 0],
            difficulty=rng.normal(size=n_items),
        )
    else:
        model_class = (
            ThreeParameterLogistic if model_name == "3PL" else FourParameterLogistic
        )
        model = model_class(n_items=n_items)
        parameters = {
            "discrimination": rng.uniform(0.8, 2.0, size=n_items),
            "difficulty": rng.normal(size=n_items),
            "guessing": np.full(n_items, 0.25),
        }
        if model_name == "4PL":
            parameters["upper"] = np.full(n_items, 0.9)
        model.set_parameters(**parameters)

    theta = rng.normal(size=(n_persons, n_factors))
    probabilities = model.probability(theta)
    responses = (rng.random(probabilities.shape) < probabilities).astype(int)
    responses[rng.random(responses.shape) < 0.05] = -1
    return model, responses, theta


class TestStandaloneStatisticsMatchFullAnalysis:
    """Standalone Q3/LD χ² agree with compute_ld_statistics on every backend.

    The standalone functions used to send any dichotomous model with
    discrimination and difficulty to the native 2PL kernels, silently dropping
    guessing, upper asymptotes and extra factors.
    """

    @pytest.fixture(autouse=True)
    def _restore_backend(self):
        import mirt

        previous = mirt.get_backend()
        yield
        mirt.set_backend(previous)

    @pytest.mark.parametrize("backend", ["auto", "numpy"])
    @pytest.mark.parametrize("model_name", ["1PL", "2PL", "3PL", "4PL", "2PL-2D"])
    def test_q3_and_chi2_match_full_analysis(self, model_name, backend):
        import mirt

        model, responses, theta = _simulated_binary_model(model_name)
        full = compute_ld_statistics(model, responses, theta=theta)

        mirt.set_backend(backend)
        q3 = compute_q3(model, responses, theta)
        chi2, _ = compute_ld_chi2(model, responses)

        assert q3.shape == (model.n_items, model.n_items)
        assert_allclose(q3, full.q3_matrix, rtol=1e-12, atol=1e-12)
        assert_allclose(chi2, full.ld_chi2_matrix, rtol=1e-12, equal_nan=True)


class TestChenThissenMarginalTables:
    """LD χ² compares observed tables with the model-implied marginal tables.

    The expected tables used to sum ``P_i(theta_hat) P_j(theta_hat)`` at EAP
    abilities. Shrunken abilities understate the item covariance, so the
    statistic grew with N and flagged most pairs of a correct model.
    """

    @staticmethod
    def _upper(matrix):
        return matrix[np.triu_indices_from(matrix, k=1)]

    def test_binary_statistics_match_r_mirt_with_missing_responses(self):
        model, responses = _r_binary_case()

        result = compute_ld_statistics(model, responses, n_quadpts=61)
        chi2, p_values = compute_ld_chi2(model, responses, n_quadpts=61)

        # R uses 61 rectangular points on [-6, 6]; the rules agree to ~1e-6.
        assert_allclose(self._upper(result.ld_chi2_matrix), _R_BINARY_X2, rtol=2e-5)
        assert_allclose(self._upper(result.g2_matrix), _R_BINARY_G2, rtol=2e-5)
        assert_allclose(self._upper(result.cramers_v_matrix), _R_BINARY_V, rtol=2e-5)
        assert_allclose(self._upper(result.chi2_p_value_matrix), _R_BINARY_P, rtol=2e-5)
        assert_allclose(chi2, result.ld_chi2_matrix, rtol=1e-12, equal_nan=True)
        assert_allclose(p_values, result.chi2_p_value_matrix, equal_nan=True)

    def test_graded_statistics_match_r_mirt_on_full_category_tables(self):
        model, responses = _r_graded_case()

        result = compute_ld_statistics(model, responses, n_quadpts=61)

        assert_allclose(self._upper(result.ld_chi2_matrix), _R_GRADED_X2, rtol=2e-5)
        assert_allclose(self._upper(result.g2_matrix), _R_GRADED_G2, rtol=2e-5)
        assert_allclose(self._upper(result.cramers_v_matrix), _R_GRADED_V, rtol=2e-5)
        assert_allclose(self._upper(result.chi2_p_value_matrix), _R_GRADED_P, rtol=2e-5)
        assert_allclose(self._upper(result.chi2_df_matrix), [4.0, 4.0, 4.0])

    def test_statistic_does_not_depend_on_abilities(self):
        model, responses = _r_graded_case()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            shifted, _ = compute_ld_chi2(
                model, responses, theta=np.full(len(responses), 2.0)
            )
        default, _ = compute_ld_chi2(model, responses)

        assert_allclose(shifted, default, equal_nan=True)
        with pytest.warns(DeprecationWarning, match="theta"):
            compute_ld_chi2(model, responses, theta=np.zeros(len(responses)))

    def test_correct_model_keeps_nominal_false_positive_rate_at_large_n(self):
        from mirt.models.dichotomous import TwoParameterLogistic

        rng = np.random.default_rng(8)
        n_items = 10
        model = TwoParameterLogistic(n_items)
        model.set_parameters(
            discrimination=rng.uniform(1.0, 2.2, n_items),
            difficulty=rng.normal(size=n_items),
        )
        model._is_fitted = True
        theta = rng.normal(size=(6000, 1))
        probabilities = model.probability(theta)
        responses = (rng.random(probabilities.shape) < probabilities).astype(int)

        start = time.perf_counter()
        result = compute_ld_statistics(model, responses)
        elapsed = time.perf_counter() - start

        # The plug-in EAP tables flagged 39 of these 45 pairs; now 2 are flagged.
        flagged = np.mean(self._upper(result.chi2_p_value_matrix) < 0.05)
        assert flagged < 0.2
        assert elapsed < 5.0


class TestQ3RawResiduals:
    """Q3 correlates the raw residuals x - E[x | theta] (Yen, 1984)."""

    @pytest.mark.parametrize("case", ["binary", "graded"])
    def test_q3_is_the_raw_residual_correlation(self, case):
        model, responses = _r_binary_case() if case == "binary" else _r_graded_case()
        theta = np.linspace(-2.0, 2.0, len(responses))[:, None]
        probabilities = np.asarray(model.probability(theta))
        expected_score = (
            probabilities
            if probabilities.ndim == 2
            else probabilities @ np.arange(probabilities.shape[2])
        )
        raw = np.where(responses >= 0, responses - expected_score, np.nan)
        n_items = responses.shape[1]
        reference = np.zeros((n_items, n_items))
        for first in range(n_items):
            for second in range(first + 1, n_items):
                valid = ~np.isnan(raw[:, first]) & ~np.isnan(raw[:, second])
                reference[first, second] = reference[second, first] = np.corrcoef(
                    raw[valid, first], raw[valid, second]
                )[0, 1]

        assert_allclose(compute_q3(model, responses, theta), reference, atol=1e-13)
        full = compute_ld_statistics(model, responses, theta=theta)
        assert_allclose(full.q3_matrix, reference, atol=1e-13)

    def test_graded_q3_matches_r_mirt(self):
        model, responses = _r_graded_case()

        q3 = compute_q3(model, responses)

        # Python and R EAP scores differ only by their quadrature rules.
        assert_allclose(q3[np.triu_indices(3, k=1)], _R_GRADED_Q3, atol=1e-5)


class TestUndefinedPairs:
    """Pairs without enough joint responses and the diagonal have no test."""

    @staticmethod
    def _planned_missing():
        model, _ = _r_binary_case()
        rng = np.random.default_rng(3)
        theta = rng.normal(size=(400, 1))
        probabilities = model.probability(theta)
        responses = (rng.random(probabilities.shape) < probabilities).astype(float)
        # Items 0 and 3 are jointly observed for only five persons.
        responses[:200, 0] = np.nan
        responses[205:, 3] = np.nan
        return model, responses

    def test_compute_ld_chi2_reports_nan_p_values(self):
        model, responses = self._planned_missing()

        chi2, p_values = compute_ld_chi2(model, responses)

        assert np.isnan(chi2[0, 3]) and np.isnan(p_values[0, 3])
        assert np.all(np.isnan(np.diag(chi2)))
        assert np.all(np.isnan(np.diag(p_values)))
        assert_allclose(np.isnan(p_values), np.isnan(chi2))

    def test_sibling_apis_share_eligibility_and_values(self):
        from mirt.utils.residuals import LD_X2

        model, responses = self._planned_missing()

        chi2, p_values = compute_ld_chi2(model, responses)
        result = compute_ld_statistics(model, responses)
        utility_chi2, utility_p = LD_X2(model, responses)

        assert_allclose(result.ld_chi2_matrix, chi2, rtol=1e-12, equal_nan=True)
        assert_allclose(result.chi2_p_value_matrix, p_values, equal_nan=True)
        assert_allclose(utility_chi2, chi2, rtol=1e-12, equal_nan=True)
        assert_allclose(utility_p, p_values, rtol=1e-12, equal_nan=True)


class TestComputeLDStatisticsValidation:
    """compute_ld_statistics validates its flagging thresholds."""

    @pytest.mark.parametrize(
        ("options", "name"),
        [
            ({"alpha": 5.0}, "alpha"),
            ({"alpha": 0.0}, "alpha"),
            ({"alpha": np.nan}, "alpha"),
            ({"q3_threshold": -1.0}, "q3_threshold"),
            ({"q3_threshold": np.inf}, "q3_threshold"),
        ],
    )
    def test_rejects_out_of_range_thresholds(self, options, name):
        from mirt.exceptions import MirtValidationError

        model, responses = _r_binary_case()
        with pytest.raises(MirtValidationError, match=name):
            compute_ld_statistics(
                model, responses, theta=np.zeros(len(responses)), **options
            )

    @pytest.mark.parametrize(
        ("options", "name"),
        [({"alpha": "0.05"}, "alpha"), ({"q3_threshold": True}, "q3_threshold")],
    )
    def test_rejects_non_numeric_thresholds(self, options, name):
        model, responses = _r_binary_case()
        with pytest.raises(TypeError, match=name):
            compute_ld_statistics(
                model, responses, theta=np.zeros(len(responses)), **options
            )


class TestDimensionAwareQuadrature:
    """LD diagnostics follow the dimension-aware EAP grid by default."""

    def test_default_grid_sizes(self):
        from mirt.diagnostics._quadrature import (
            default_diagnostic_quadpts,
            resolve_diagnostic_quadpts,
        )
        from mirt.exceptions import MirtValidationError

        assert [default_diagnostic_quadpts(n) for n in range(1, 11)] == [
            49,
            49,
            21,
            9,
            7,
            5,
            5,
            4,
            3,
            3,
        ]
        assert resolve_diagnostic_quadpts(None, 4) == 9
        assert resolve_diagnostic_quadpts(13, 4) == 13
        for value in (1, 2.5, True):
            with pytest.raises(MirtValidationError, match="n_quadpts"):
                resolve_diagnostic_quadpts(value, 1)

    def test_four_factor_default_uses_nine_points(self):
        from mirt.models.dichotomous import TwoParameterLogistic

        rng = np.random.default_rng(0)
        n_items, n_factors = 8, 4
        slopes = np.zeros((n_items, n_factors))
        for factor in range(n_factors):
            slopes[2 * factor : 2 * factor + 2, factor] = rng.uniform(1.0, 2.0, 2)
        model = TwoParameterLogistic(n_items, n_factors=n_factors)
        model.set_parameters(discrimination=slopes, difficulty=np.zeros(n_items))
        model._is_fitted = True
        theta = rng.normal(size=(80, n_factors))
        probabilities = model.probability(theta)
        responses = (rng.random(probabilities.shape) < probabilities).astype(int)

        start = time.perf_counter()
        default = compute_ld_statistics(model, responses)
        elapsed = time.perf_counter() - start
        explicit = compute_ld_statistics(model, responses, n_quadpts=9)

        assert_allclose(default.q3_matrix, explicit.q3_matrix)
        assert_allclose(default.ld_chi2_matrix, explicit.ld_chi2_matrix, equal_nan=True)
        assert elapsed < 2.0
