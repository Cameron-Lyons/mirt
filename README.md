# mirt

**Multidimensional Item Response Theory for Python**

A comprehensive Python implementation of Item Response Theory (IRT) models with a high-performance Rust backend, inspired by R's [mirt](https://github.com/philchalmers/mirt) package.

## Features

### Core IRT Models
- **Dichotomous**: 1PL (Rasch), 2PL, 3PL, 4PL, 5PL, complementary and negative log-log
- **Polytomous**: GRM, GPCM, PCM, NRM, rating scale (RSM) and graded rating scale (GRSM)
- **Mixed formats**: One family per item, e.g. 3PL multiple choice with GRM constructed response
- **Multidimensional**: Exploratory and confirmatory MIRT with estimated factor correlations
- **Bifactor**: Bifactor and hierarchical models, with dimension-reduction EM for any number of specific factors

### Advanced Models
- **Cognitive Diagnostic**: DINA, DINO, G-DINA
- **Testlet**: Random effects for item bundles
- **Nested Logit**: Keyed multiple-choice items with informative distractors
- **Mixture IRT**: Latent class IRT models
- **Zero-Inflated**: ZI-2PL, ZI-3PL, Hurdle IRT
- **Unfolding**: GGUM, Ideal Point, Hyperbolic Cosine
- **Nonparametric**: Monotonic spline and weighted kernel-smoothed IRFs
- **Many-Facet Rasch**: Rater, task, and criterion facets by marginal maximum likelihood
- **Knowledge tracing**: Bayesian knowledge tracing by Baum-Welch EM or Gibbs sampling, plus longitudinal and growth models
- **Network Psychometrics**: Ising and sparse Gaussian graphical models

### Estimation Methods
- **EM Algorithm**: Gauss-Hermite quadrature (with Rust acceleration) and optional SQUAREM acceleration
- **Estimation controls**: Starting values, fixed parameters, item priors (Bayes modal EM), and R-style model syntax
- **Standard errors**: Observed-information (Louis/Oakes) standard errors and parameter covariance by default for unidimensional 1PL-4PL, graded, partial credit, rating scale, and mixed-format EM fits; cross-product and sandwich alternatives
- **GVEM**: Gaussian Variational EM for fast high-dimensional estimation
- **Sparse Bayesian**: Spike-slab LASSO for automatic structure discovery
- **MHRM**: Metropolis-Hastings Robbins-Monro (Cai, 2010) for dichotomous and polytomous items
- **MCMC**: Gibbs sampling for Bayesian estimation
- **MCEM/QMCEM**: Monte Carlo EM for high dimensions, with automatic MCEM sample-size growth

### Computerized Adaptive Testing (CAT)
- Item selection: MFI, MEI, KL divergence, a-stratified, Urry, shadow tests
- Stopping rules: SE threshold, max items, classification, estimate or SE change, minimum information, predicted SE reduction
- Exposure control: Sympson-Hetter, randomesque, progressive
- Content balancing: Blueprint constraints, met exactly by shadow tests
- Fixed and parallel form assembly: Exact optimization with content, security, cost, item bundles, reuse, and overlap constraints
- Simulation studies: Vectorized lock-step batch simulation and catR-style reports (bias, RMSE, exposure, overlap)
- **MCAT**: Multidimensional CAT with D-optimality and trace criteria

### Diagnostics & DIF
- **Item fit**: Infit, outfit (with standardized z values), S-X2, X2/G2, and PV-Q1 with multiple-testing control
- **Person fit**: Zh, lz, infit/outfit
- **Model fit**: M2, RMSEA, CFI, TLI, SRMSR
- **DIF analysis**: Multiple-group likelihood ratio (drop/add schemes, anchor selection), linked Wald and Lord, Raju
- **GRDIF**: Multi-group residual DIF with robust scaling and multiplicity control
- **DTF/DRF**: Differential test/response functioning
- **SIBTEST**: Simultaneous item bias test
- **Local dependence**: Q3, chi-square residuals, and pairwise multiplicity control

### Additional Features
- Custom dichotomous, ordinal, multidimensional, and latent group models
- Multiple group analysis with invariance testing
- Bootstrap standard errors, confidence intervals, and likelihood-ratio tests
- Plausible values for population inference, with population or latent-regression priors
- Missing data imputation with calibrated-model reuse and joint posterior ability draws
- Built-in sample datasets
- Simulation from fitted dichotomous and polytomous models
- Plotting (ICC, information, Wright maps, DIF)
- DataFrame input and output (pandas or polars): column names become item names, and NaN marks missing responses when fitting
- Fixed-item parameter calibration (FIPC), IRT linking (including polytomous chain linking), and true-score, observed-score, and kernel equating
- **Vertical scaling**: Grade-level linking with growth constraints
- Reliable Change Index (RCI) for clinical significance
- Profile-likelihood confidence intervals
- Posterior parameter sampling
- **Result objects**: Validated uncertainty, confidence intervals, and portable exports that reload with `FitResult.from_json`
- **HTML reports**: Safe standalone summaries with optional embedded plots

## Installation

```bash
pip install mirt
```

With optional dependencies:
```bash
pip install mirt[pandas]
pip install mirt[polars]
pip install mirt[dev]
```

For plotting support:
```bash
pip install "mirt[plot]"
```

## Quick Start

```python
import mirt

dataset = mirt.load_dataset("LSAT7")
responses = dataset["data"]

result = mirt.fit_mirt(responses, model="2PL")
print(result.summary())

scores = mirt.fscores(result, responses, method="EAP")
print(scores.to_dataframe().head())
```

## Examples

### Simulating Data

```python
import mirt
import numpy as np

responses = mirt.simdata(model="2PL", n_persons=500, n_items=20, seed=42)

a = np.random.lognormal(0, 0.3, size=20)
b = np.random.normal(0, 1, size=20)
responses = mirt.simdata(model="2PL", discrimination=a, difficulty=b, n_persons=1000)

likert_data = mirt.simdata(model="GRM", n_categories=5, n_persons=500, n_items=15)

pcm_params = mirt.generate_item_parameters(
    n_items=15, model="PCM", n_categories=5, seed=42
)
pcm_data = mirt.simdata(
    model="PCM", n_persons=500, n_items=15, n_categories=5, **pcm_params
)

nrm_params = mirt.generate_item_parameters(
    n_items=10, model="NRM", n_categories=4, n_factors=2, seed=42
)
nrm_data = mirt.simdata(
    model="NRM", n_persons=500, n_items=10, n_categories=4,
    n_factors=2, **nrm_params
)

zi_params = mirt.generate_item_parameters(n_items=20, model="ZI-2PL", seed=42)
zi_data, structural_zeros = mirt.simdata(
    model="ZI-2PL", n_persons=500, n_items=20,
    return_structural_zeros=True, **zi_params, seed=43
)
```

### Fitting Models

```python
result_1pl = mirt.fit_mirt(responses, model="1PL")
result_2pl = mirt.fit_mirt(responses, model="2PL")
result_3pl = mirt.fit_mirt(responses, model="3PL", accelerate="squarem")

result_grm = mirt.fit_mirt(likert_data, model="GRM", n_categories=5)
result_gpcm = mirt.fit_mirt(likert_data, model="GPCM", n_categories=5)
result_rsm = mirt.EMEstimator().fit(
    mirt.RatingScaleModel(n_items=15, n_categories=5), likert_data
)

result_mirt = mirt.fit_mirt(responses, model="2PL", n_factors=2)

# Observed-information standard errors and the parameter covariance.
print(result_2pl.se_method, result_2pl.vcov.shape)
print(mirt.wald(result_2pl, param_indices=[0], constraint_values=[1.0]))

# Simulate a new sample from a fitted model.
new_likert = mirt.simdata(result_grm, n_persons=300, seed=1)

result = result_2pl  # used by the examples below
```

### Starting Values, Fixed Parameters, and Priors

```python
from mirt.estimation.priors import BetaPrior

anchored = np.zeros(20, dtype=bool)
anchored[:5] = True  # hold the first five slopes at their starting values

result_map = mirt.fit_mirt(
    responses,
    model="3PL",
    start_values={"discrimination": np.where(anchored, 1.2, 1.0)},
    fixed={"discrimination": anchored},
    priors={"guessing": BetaPrior(5, 17)},  # Bayes modal (MAP) EM
)
print(result_map.log_likelihood, result_map.log_posterior)
```

### Confirmatory Models with Model Syntax

```python
rng = np.random.default_rng(1)
theta_2d = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=1000)
slopes = np.zeros((20, 2))
slopes[:10, 0] = slopes[10:, 1] = 1.5
cfa_data = mirt.simdata(
    model="2PL", theta=theta_2d, discrimination=slopes, n_factors=2, seed=1
)

spec = mirt.mirt_model("""
    F1 = 1-10
    F2 = 11-20
    COV = F1*F2
""")
result_cfa = mirt.fit_mirt(cfa_data, model="2PL", spec=spec, n_quadpts=15)
print(result_cfa.factor_correlation)  # estimated correlation near 0.5
```

### Mixed Item Formats and Bifactor Models

```python
ability = rng.standard_normal(1000)
mixed_data = np.column_stack([
    mirt.simdata(model="3PL", theta=ability, n_items=15, seed=2),
    mirt.simdata(model="GRM", theta=ability, n_items=5, n_categories=4, seed=3),
])
result_mixed = mirt.fit_mirt(mixed_data, model=["3PL"] * 15 + ["GRM"] * 5)
print(result_mixed.coef())  # one row per item

# Bifactor: each specific factor is integrated jointly with the general factor.
specific = np.repeat([0, 1, 2, 3], 5)
loadings = np.zeros((20, 5))
loadings[:, 0] = 1.4
loadings[np.arange(20), 1 + specific] = 1.0
bifactor_data = mirt.simdata(
    model="2PL", theta=rng.standard_normal((1000, 5)),
    discrimination=loadings, n_factors=5, seed=4,
)
result_bf = mirt.bfactor(bifactor_data, specific)
print(result_bf.model.general_loadings)
```

### Weighted Nonparametric Calibration

```python
from mirt.models import KernelSmoothingModel

theta = np.linspace(-3, 3, 500)
kernel_data = mirt.simdata(model="2PL", theta=theta, n_items=20, seed=7)
person_weight = np.linspace(0.5, 1.5, theta.size)

kernel_model = KernelSmoothingModel(n_items=kernel_data.shape[1]).calibrate(
    kernel_data, theta, sample_weight=person_weight
)
smoothed_curves = kernel_model.probability(np.linspace(-3, 3, 121))
```

### Person Scoring

```python
eap = mirt.fscores(result, responses, method="EAP")
map_scores = mirt.fscores(result, responses, method="MAP")
ml = mirt.fscores(result, responses, method="ML")

print(eap.theta)
print(eap.standard_error)

# Convert latent scores to a T-score scale while propagating uncertainty.
t_scores = eap.linear_transform(multiplier=10, offset=50)
percentile_ranks = t_scores.normal_percentile_ranks(
    reference_mean=50,
    reference_sd=10,
)

# EAP ability for every attainable sum score, with observed and expected counts.
from mirt.scoring import eapsum_table

sum_score_table = eapsum_table(result, responses)
```

### Posterior Ability Distributions

```python
posterior = mirt.ability_posterior(result, responses, batch_size=512)
lower, upper = posterior.credible_intervals(level=0.95)
percentiles = posterior.quantile([0.1, 0.5, 0.9])
probability_above = posterior.classification_probabilities(cut_score=0.0)
plausible_values = posterior.sample(n_draws=10, seed=42)
```

Posterior summaries use bounded temporary memory. Multiple quantiles share the
same cumulative distributions, and sampling reuses the stored joint posterior
without evaluating the model again. Draws have shape
`(n_persons, n_factors, n_draws)` and preserve dependence between factors.

### Diagnostics

```python
item_fit = mirt.itemfit(
    result,
    responses,
    statistics=["infit", "outfit", "S_X2"],
    p_adjust="fdr_bh",
)
print(item_fit)

# Standardized mean squares and ability-grouped (Q1-type) item fit.
grouped_fit = mirt.itemfit(
    result,
    responses,
    statistics=["z_infit", "z_outfit", "X2", "PV_Q1"],
    n_plausible=20,
    seed=1,
)

person_fit = mirt.personfit(result, responses)
aberrant = np.flatnonzero(np.asarray(person_fit["Zh"]) < -2)

fit_indices = mirt.compute_fit_indices(result.model, responses)
print(fit_indices)

from mirt.diagnostics import compute_ld_statistics

ld = compute_ld_statistics(
    result.model,
    responses,
    p_adjust="fdr_bh",
)
print(ld.chi2_adjusted_p_value_matrix)

results = [result_1pl, result_2pl, result_3pl]
comparison = mirt.compare_models(results)
```

### DIF Analysis

```python
groups = np.array([0] * 500 + [1] * 500)

dif_lr = mirt.dif(responses, groups, method="likelihood_ratio")

dif_wald = mirt.dif(responses, groups, method="wald")
dif_lord = mirt.dif(responses, groups, method="lord")
dif_raju = mirt.dif(responses, groups, method="raju")

# Multiple-group likelihood-ratio DIF with data-driven anchors (R mirt::DIF).
from mirt.multigroup import multigroup_dif, select_dif_anchors

anchors = select_dif_anchors(responses, groups, model="2PL", n_anchors=4)
dif_table = multigroup_dif(
    responses, groups, model="2PL", scheme="add", anchors=anchors
)

from mirt.diagnostics.dif import compute_grdif

groups_multi = np.array(["A"] * 400 + ["B"] * 300 + ["C"] * 300)
grdif_result = compute_grdif(
    responses, groups_multi,
    model="2PL",
    scaling_method="mad",
    p_adjust="fdr_bh",
)
print(f"Flagged items: {np.where(grdif_result['flagged_rs'])[0]}")
```

### Multiple Group Analysis

```python
from mirt.multigroup import fit_multigroup, compare_invariance

mg_result = fit_multigroup(responses, groups, model="2PL", invariance="metric")

invariance = compare_invariance(responses, groups, model="2PL", verbose=True)
```

### Computerized Adaptive Testing

```python
from mirt.cat import CATEngine, summarize_cat_simulation

cat = CATEngine(result.model, se_threshold=0.3, max_items=20)

# Supported configurations simulate all examinees in vectorized lock step.
true_thetas = np.linspace(-2, 2, 11)
sim_results = cat.run_batch_simulation(true_thetas, n_replications=100)
report = summarize_cat_simulation(
    sim_results, true_thetas, n_items=result.model.n_items, n_replications=100
)
print(report.summary())  # bias, RMSE, test length, exposure, overlap

state = cat.get_current_state()
while not state.is_complete:
    item = state.next_item
    response = get_examinee_response(item)
    state = cat.administer_item(response)

final = cat.get_result()
print(final.summary())

# Shadow tests meet the content blueprint exactly; the SE rule is paired
# with a rule that stops once further items barely reduce the SE.
from mirt.cat import CombinedStop, ContentArea, ContentBlueprint
from mirt.cat import PredictedSEReductionStop, ShadowTestSelection, StandardErrorStop

blueprint = ContentBlueprint([
    ContentArea("Algebra", items=set(range(10)), min_items=5, max_items=5),
    ContentArea("Geometry", items=set(range(10, 20)), min_items=5, max_items=5),
])
shadow_cat = CATEngine(
    result.model,
    item_selection=ShadowTestSelection(blueprint=blueprint),
    max_items=10,
    min_items=10,
)
shadow_result = shadow_cat.run_simulation(true_theta=0.5)

stopping = CombinedStop([
    StandardErrorStop(0.3),
    PredictedSEReductionStop(result.model, min_reduction=0.01),
])
early_stop_cat = CATEngine(result.model, stopping_rule=stopping, max_items=20)

from mirt.cat import MCATEngine

mcat = MCATEngine(
    result_mirt.model,
    item_selection="D-optimality",
    max_items=15,
)
mcat_result = mcat.run_simulation(true_theta=np.array([0.5, -0.3]))
print(f"Estimated theta: {mcat_result.theta}")
print(f"Covariance: {mcat_result.covariance}")
```

### Advanced Models

```python
from mirt import fit_cdm
q_matrix = np.tile([[1, 0], [1, 1], [0, 1], [1, 1]], (5, 1))  # one row per item
cdm_model, profile_posteriors = fit_cdm(responses, q_matrix, model="DINA")

from mirt import fit_mixture_irt
mix_model, class_posteriors = fit_mixture_irt(
    responses, n_classes=2, base_model="2PL"
)

from mirt import TestletModel, create_testlet_structure
testlet_struct = create_testlet_structure(n_items=20, testlet_sizes=[5, 5, 5, 5])

# Bayesian knowledge tracing by maximum likelihood (Baum-Welch EM).
from mirt.models import BKTModel

practice, skills, _ = BKTModel(n_skills=3).simulate(
    n_persons=500, n_trials_per_skill=10, seed=1
)
bkt = mirt.fit_bkt_em(practice, skills, n_starts=4, seed=0)
print(bkt.model.p_learn, bkt.converged)

# Many-facet Rasch model: rater severities by marginal maximum likelihood.
from mirt.models import Facet, PolytomousMFRM

raters = rng.integers(0, 4, size=(500, 6))
truth = PolytomousMFRM(6, 5, [Facet("rater", 4)])
truth.set_facet_parameters("rater", np.array([-0.5, -0.1, 0.2, 0.4]))
ratings = truth.simulate(rng.standard_normal(500), {"rater": raters}, seed=2)

mfrm = mirt.fit_mfrm(
    PolytomousMFRM(6, 5, [Facet("rater", 4)]), ratings, {"rater": raters}
)
print(mfrm.summary())  # measures, standard errors, infit and outfit
```

### Custom Item Models

```python
import numpy as np

from mirt import CustomItemModel, create_item_type

def adjacent_categories(theta, shift):
    weights = np.column_stack((
        np.ones_like(theta),
        np.exp(theta - shift),
        np.exp(2 * (theta - shift)),
    ))
    return weights / weights.sum(axis=1, keepdims=True)

spec = create_item_type(
    "AdjacentCategories",
    adjacent_categories,
    par_bounds={"shift": (-4, 4)},
    par_defaults={"shift": 0},
    n_categories=3,
)
model = CustomItemModel(n_items=10, item_type=spec)
probabilities = model.probability(np.linspace(-3, 3, 61))
```

### Exploratory Factor Analysis with Automatic Structure Discovery

```python
from mirt import TwoParameterLogistic
from mirt.estimation import SparseBayesianEstimator, GVEMEstimator

model = TwoParameterLogistic(n_items=20, n_factors=5)
estimator = SparseBayesianEstimator(k_max=5, lambda_0=0.04, lambda_1=1.0)
sparse_result = estimator.fit(model, responses)

print(f"Effective dimensions: {sparse_result.effective_dimensionality}")
print(f"Sparsity ratio: {1 - sparse_result.sparsity_pattern.mean():.1%}")
print(sparse_result.loading_table())

estimator = GVEMEstimator(max_iter=200, tol=1e-4)
gvem_result = estimator.fit(model, responses)
```

### Test Equating & Calibration

```python
from mirt.equating import irt_kernel_equating, kernel_equating, link

# form_a_model and form_b_model are calibrations of two forms sharing items 0-4.
linking = link(
    form_a_model,
    form_b_model,
    anchor_items_old=[0, 1, 2, 3, 4],
    anchor_items_new=[0, 1, 2, 3, 4],
    method="stocking_lord",
)
A, B = linking.constants.A, linking.constants.B
print(f"Scale transformation: theta_old = {A:.3f} * theta_new + {B:.3f}")

# IRT observed-score kernel equating on the linked scale, and Gaussian kernel
# equating of observed sum-score frequencies with log-linear presmoothing.
irt_kernel = irt_kernel_equating(form_a_model, form_b_model, linking_result=linking)
kernel = kernel_equating(
    frequencies_a,
    frequencies_b,
    presmoothing=3,
    n_old=int(frequencies_a.sum()),
    n_new=int(frequencies_b.sum()),
)
print(kernel.new_scores, kernel.standard_errors)

# Fixed-item calibration (FIPC): new items and the calibration population are
# estimated on the scale of anchor items whose parameters stay fixed.
calibration = mirt.fixed_item_calibration(
    combined_responses,
    mirt.TwoParameterLogistic(n_items=combined_responses.shape[1]),
    anchor_items=[0, 1, 2, 3, 4],
    anchor_parameters=anchor_model,  # calibrated model of the five anchors
)
print(calibration.latent_mean, calibration.new_item_parameters["difficulty"])

q3_matrix = mirt.Q3(result.model, responses, eap.theta)
resid = mirt.residuals(result.model, responses, eap.theta)
print(f"Max Q3 (off-diagonal): {np.max(np.abs(np.triu(q3_matrix, 1))):.3f}")
```

### Vertical Scaling

```python
from mirt.equating import vertical_scale, GradeData, compute_vertical_diagnostics

grade_data = [
    GradeData("Grade 3", responses_g3, anchor_items_above=[0, 1, 2, 3, 4]),
    GradeData("Grade 4", responses_g4, anchor_items_below=[10, 11, 12, 13, 14],
              anchor_items_above=[0, 1, 2, 3, 4]),
    GradeData("Grade 5", responses_g5, anchor_items_below=[10, 11, 12, 13, 14]),
]

vertical = vertical_scale(
    grade_data,
    method="chain",
    enforce_monotonicity=True,
)

print(f"Grade means: {vertical.grade_means}")
print(f"Growth curve: {vertical.growth_curve}")

diagnostics = compute_vertical_diagnostics(vertical, grade_data)
print(f"Grade separation (effect sizes): {diagnostics.grade_separation}")
```

### Bootstrap Tests, Plausible Values, and Portable Results

```python
# Parametric bootstrap likelihood-ratio test for a boundary comparison.
lr_test = mirt.bootstrap_lr(result_2pl, result_3pl, responses, n_bootstrap=100, seed=42)
print(lr_test.p_value, lr_test.asymptotic_p_value)

# Plausible values under the population estimated by fixed-item calibration.
plausible = mirt.generate_plausible_values(
    calibration.model,
    combined_responses,
    n_plausible=10,
    prior_mean=calibration.latent_mean,
    prior_cov=calibration.latent_cov,
    seed=1,
)

# DataFrame input: column names become item names and NaN/NA marks missing.
import pandas as pd

frame = pd.DataFrame(likert_data, columns=[f"Q{i + 1}" for i in range(15)])
frame = frame.astype("Float64")
frame.iloc[0, 0] = pd.NA
result_df = mirt.fit_mirt(frame, model="GRM")

# Save a calibration as JSON and reload it for scoring.
restored = mirt.FitResult.from_json(result_df.to_json())
restored_scores = mirt.fscores(restored, likert_data)
```

### Plotting

```python
from mirt import (
    plot_category_curves,
    plot_icc,
    plot_information,
    plot_person_item_map,
)

plot_icc(result.model, item_idx=[0, 1, 2])

plot_information(result.model)

# Polytomous category response curves
plot_category_curves(result_grm.model, item_idx=0)

plot_person_item_map(result.model, eap.theta)
```

## Supported Models

### Dichotomous Models

| Model | Description | Parameters |
|-------|-------------|------------|
| 1PL/Rasch | One-parameter logistic | difficulty (b) |
| 2PL | Two-parameter logistic | discrimination (a), difficulty (b) |
| 3PL | Three-parameter logistic | a, b, guessing (c) |
| 4PL | Four-parameter logistic | a, b, c, upper asymptote (d) |
| 5PL | Five-parameter logistic | a, b, c, d, asymmetry (e) |

### Polytomous Models

| Model | Description | Use Case |
|-------|-------------|----------|
| GRM | Graded Response Model | Ordered categories (Likert) |
| GPCM | Generalized Partial Credit | Partial credit scoring |
| PCM | Partial Credit Model | Rasch for polytomous |
| RSM | Rating Scale Model | Rasch rating scale with thresholds shared by all items |
| GRSM | Graded Rating Scale Model | Graded items sharing one slope and threshold set |
| NRM | Nominal Response Model | Unordered categories |
| 2PL/3PL/4PL-NRM | Nested Logit | Keyed multiple choice with distractor information |

### Advanced Models

| Model | Description |
|-------|-------------|
| MIRT | Multidimensional IRT |
| Bifactor | General + specific factors |
| Mixed format | A different item family for each item on one trait |
| DINA/DINO | Cognitive diagnostic |
| Testlet | Local dependence modeling |
| Nested Logit | Keyed response and conditional distractor modeling |
| Mixture IRT | Latent class IRT |
| GGUM | Generalized graded unfolding |
| MFRM | Many-facet Rasch models for rater and task effects |
| BKT | Bayesian knowledge tracing of per-skill mastery |

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `fit_mirt()` | Fit IRT models |
| `bfactor()` | Full-information bifactor models with dimension reduction |
| `mirt_model()` | Parse R `mirt.model`-style syntax for confirmatory models |
| `fscores()` | Person ability estimation |
| `simdata()` | Simulate response data, including from fitted models |
| `itemfit()` | Item fit statistics |
| `personfit()` | Person fit statistics |
| `dif()` | DIF analysis |
| `multigroup_dif()` | Multiple-group likelihood-ratio DIF with drop/add schemes |
| `fit_mfrm()` | Many-facet Rasch models by marginal maximum likelihood |
| `fit_bkt_em()` | Bayesian knowledge tracing by Baum-Welch EM |
| `load_dataset()` | Load sample datasets |

### Estimator Classes

| Class | Description |
|-------|-------------|
| `EMEstimator` | Standard EM with Gauss-Hermite quadrature, item priors, and SQUAREM |
| `BifactorEMEstimator` | Bifactor EM on two-dimensional grids per specific factor |
| `MixedFormatEMEstimator` | EM for tests that mix item families |
| `MHRMEstimator` | Metropolis-Hastings Robbins-Monro |
| `GVEMEstimator` | Gaussian Variational EM (fast, high-dimensional) |
| `SparseBayesianEstimator` | Spike-slab LASSO for sparse structure discovery |
| `MCEMEstimator` | Monte Carlo EM for high dimensions |
| `QMCEMEstimator` | Quasi-Monte Carlo EM |
| `StochasticEMEstimator` | Stochastic EM |

### Diagnostic Functions

| Function | Description |
|----------|-------------|
| `compute_fit_indices()` | M2/M2* score moments, RMSEA, CFI, TLI, SRMSR |
| `compare_models()` | AIC/BIC comparison |
| `anova_irt()` | Likelihood ratio tests |
| `compute_dtf()` | Differential test functioning |
| `compute_drf()` | Differential response functioning |
| `sibtest()` | SIBTEST DIF detection |
| `compute_grdif()` | Multi-group GRDIF with robust scaling |
| `select_dif_anchors()` | DIF-free anchor items by iterative or ranked likelihood-ratio tests |
| `vertical_scale()` | Vertical scaling for grade linking |
| `mirt.diagnostics.psis_loo()` | PSIS-LOO with optional threaded observation smoothing |
| `mirt.diagnostics.posterior_predictive_checks()` | Shared-simulation suites for built-in or custom fit statistics |

### Utility Functions

| Function | Description |
|----------|-------------|
| `bootstrap_se()` | Bootstrap standard errors with optional process workers |
| `bootstrap_ci()` | Percentile, basic, and BCa intervals with parallel replicate and jackknife fits |
| `parametric_bootstrap()` | Deterministic model-based bootstrap with optional process workers |
| `bootstrap_lr()` | Parametric bootstrap likelihood-ratio test for nested models |
| `generate_plausible_values()` | Plausible values, optionally under a population prior |
| `combine_plausible_values()` | Validated scalar or vector combining with Rubin uncertainty |
| `plausible_value_statistics()` | Mean, variance, SD, or percentiles for one or all latent factors |
| `plausible_value_regression()` | Validated ordinary or weighted regression with combined uncertainty |
| `cross_validate()` | Validated K-fold, stratified, group-aware, and leave-one-out evaluation with optional process parallelism and shared per-fold ability estimates |
| `cv_select_lambda()` | Held-out regularization selection with optional process-parallel fold evaluation |
| `impute_responses()` | Missing data imputation |
| `itemstats()` | Item distributions, modes, entropy, and effective category counts |
| `missing_patterns()` | Frequency-ranked missing-response pattern analysis |
| `gen_random_pars()` | Valid random starting values that preserve model constraints |
| `multi_start_fit()` | Repeated fitting with deterministic best-fit selection |
| `calc_null()` | Independence and pooled-intercept baseline fit statistics |
| `fit_models()` | Validated sequential or parallel model comparison |
| `fit_model_grid()` | Hyperparameter grids with retained failure details |
| `set_dataframe_backend()` | Choose pandas/polars or restore automatic selection |
| `get_dataframe_backend()` | Inspect the active DataFrame backend |
| `residuals()` | Model residuals (raw, standardized, Pearson, deviance) |
| `Q3()` | Yen's Q3 local dependence statistic |
| `LD_X2()` | Chen & Thissen LD chi-square |
| `fixed_item_calibration()` | Fixed-item parameter calibration (FIPC) for any item family |
| `fixed_calib()` | Legacy 2PL fixed-item calibration |
| `link()` / `chain_link()` | IRT linking of separately calibrated forms |
| `kernel_equating()` / `irt_kernel_equating()` | Gaussian kernel equating of observed or IRT score distributions |
| `equate()` | Deprecated; use `link()` |
| `RCI()` | Reliable Change Index for clinical significance |
| `PLCI()` | Profile-likelihood confidence intervals |
| `wald()` | Wald tests using a fit's parameter covariance |
| `draw_parameters()` | Draw item parameters from a fit's asymptotic covariance |
| `posterior_summary()` | Summarize sampled parameter uncertainty |
| `sample_expected_scores()` | Propagate parameter uncertainty to test, subtest, or item scores |
| `randef()` / `fixef()` | Random/fixed effects from mixed models |
| `predict_mixed()` | Response probabilities from abilities or person covariates |
| `conditional_effects()` / `shrinkage_estimates()` | Mixed-model effect and reliability summaries |
| `empirical_plot()` / `empirical_rmsea()` | Binned binary and polytomous empirical-fit diagnostics |
| `itemGAM()` | Kernel-smoothed observed-versus-expected item scores |
| `rotate_loadings()` | Varimax, quartimax, equamax, oblimin, promax, and geomin rotations |

### Data Transformation Functions

| Function | Description |
|----------|-------------|
| `key2binary()` | Score multiple choice with answer key |
| `poly2dich()` | Convert polytomous to dichotomous |
| `reverse_score()` | Reverse score items |
| `expand_table()` | Expand frequency table to response matrix |
| `collapse_table()` | Collapse responses to frequency table |
| `collapse_patterns()` | Collapse duplicate response patterns for efficient estimation |
| `collapse_with_groups()` | Collapse response patterns independently within groups |
| `recode_responses()` | Recode response values |

### Information Functions

| Function | Description |
|----------|-------------|
| `testinfo()` | Test information function |
| `iteminfo()` | Item information function |
| `areainfo()` | Area under information curve |
| `expected_score()` | Expected score at theta |
| `gen_difficulty()` | Generalized difficulty index |
| `theta_for_score()` | Find theta for target score |
| `information_intervals()` | Find item or test ability ranges meeting an information target |

## Comparison with R mirt

| Feature | R mirt | Python mirt |
|---------|--------|-------------|
| Dichotomous models | 1PL-4PL | 1PL-4PL |
| Polytomous models | GRM, GPCM, PCM, NRM | GRM, GPCM, PCM, NRM |
| Mixed item types | `itemtype` vector | `fit_mirt(model=[...])` |
| Multidimensional | Full support | Full support |
| Model syntax | `mirt.model()` | `mirt_model()` (FIXED, START, PRIOR, COV) |
| Bifactor | Yes | Yes (`bfactor()` with dimension reduction) |
| Cognitive diagnostic | mirtCAT separate | Built-in (DINA, DINO) |
| Estimation | EM, MHRM, MCMC | EM (with SQUAREM), GVEM, Sparse Bayesian, MHRM, MCMC |
| Automatic structure discovery | No | Yes (spike-slab LASSO) |
| CAT | mirtCAT package | Built-in (unidimensional, MCAT, shadow tests) |
| DIF | Yes | Yes (multiple-group LR, linked Wald and Lord, Raju, GRDIF) |
| Multiple groups | Full support | Full support |
| Equating | equateIRT/kequate packages | Built-in (linking, true-score, observed-score, kernel) |
| Vertical scaling | plink package | Built-in |
| HTML reports | No | Built-in |
| Rust acceleration | No | Yes (see below) |

## Rust Acceleration

When the Rust backend is available (automatically built during installation), the following operations are accelerated with parallel processing:

| Category | Accelerated Operations |
|----------|------------------------|
| **Likelihood** | Log-likelihood computation for 2PL, 3PL, and multidimensional models |
| **EM Algorithm** | E-step (posterior computation), M-step (Newton-Raphson optimization), full EM fitting |
| **Multigroup** | E-step for all models (2PL, 3PL, GRM, GPCM, NRM), expected counts |
| **Scoring** | EAP scores, WLE scores, Lord-Wingersky recursion for sum scores |
| **Diagnostics** | Q3 matrix, LD chi-square, infit/outfit statistics, standardized residuals |
| **Calibration** | Fixed-item calibration EM algorithm (`fixed_calib`) |
| **SIBTEST** | Beta statistic computation, all-items SIBTEST |
| **CAT** | Item information, item selection, EAP updates, batch simulation |
| **Simulation** | Response generation for 2PL/3PL, GRM, GPCM |
| **Bootstrap** | Index generation, resampling, parallel bootstrap fitting |
| **Plausible Values** | Posterior sampling, MCMC generation |
| **MCMC** | Gibbs sampling for 2PL, MHRM estimation |

### Fallback contract

Rust wrappers declare one of four modes (see each module's `FALLBACK_MODE`):

| Mode | Behavior when Rust is unavailable or disabled |
|------|-----------------------------------------------|
| **numpy** | Pure NumPy implementation runs automatically |
| **optional** | Returns `None`; the public caller supplies a Python path |
| **required** | Raises `RuntimeError` (accelerated-only entry point; use the public Python estimator instead) |
| **mixed** | Module contains more than one of the modes above |

Most hot paths (likelihood, E-step, scoring, diagnostics, simulation) are **numpy**. A few full-fit helpers such as `em_fit_2pl` are **required** — `fit_mirt(..., estimation="EM")` still works without Rust via `EMEstimator`.

Disable Rust globally with:

```python
import mirt
mirt.set_backend("numpy")
```

Per-call `use_rust=False` also disables Rust for that call. `mirt.should_use_rust()` reports the effective decision.

The Rust backend provides significant speedups for large datasets (1000+ persons) due to:

- **Rayon parallelization**: Computation across persons or items runs in parallel
- **SIMD optimizations**: Vectorized arithmetic where available
- **Memory efficiency**: Reduced allocations compared to NumPy broadcasting

To check if Rust acceleration is available:
```python
import mirt
print(mirt.get_backend_info())
print(f"Rust extension: {mirt.is_rust_available()}")
```

## Requirements

### Core Dependencies (always required)
- Python >= 3.11
- numpy >= 1.24
- scipy >= 1.9

### Optional Dependencies

| Package | Purpose | Installation |
|---------|---------|--------------|
| **matplotlib** | Plotting (ICC, category and information curves, Wright maps, DIF) | `pip install "mirt[plot]"` |
| **pandas** | DataFrame input and result output | `pip install mirt[pandas]` |
| **polars** | DataFrame input and output (faster, preferred when both installed) | `pip install mirt[polars]` |

When neither pandas nor polars is installed, functions that return DataFrames will raise an `ImportError` with installation instructions. Plotting functions similarly require matplotlib.

To set your preferred DataFrame backend explicitly:
```python
import mirt
mirt.set_dataframe_backend("pandas")
```

Restore automatic selection at any time with
`mirt.set_dataframe_backend("auto")` (or `None`). Use
`mirt.get_dataframe_backend()` to inspect the active backend.

## Development

```bash
git clone https://github.com/Cameron-Lyons/mirt.git
cd mirt
uv venv
uv pip install -e ".[dev]"

uv run maturin develop --release

uv run pytest

uv run mypy src/mirt

uv run ruff format src tests
uv run ruff check src tests

uv run pytest -m slow

uv run pytest tests/test_performance_smoke.py

uv run python benchmarks/run_benchmarks.py
```

The benchmark runner supports warmups, individual suites, structured JSON output,
and regression-sensitive baseline comparisons without extra dependencies:

```bash
# Record a reusable baseline for all workloads.
uv run python benchmarks/run_benchmarks.py \
  --repeats 5 --warmups 1 --json benchmark-baseline.json

# Run only scoring and fail if its median slows by more than 10%.
uv run python benchmarks/run_benchmarks.py \
  --suite scoring --repeats 5 --warmups 1 \
  --baseline benchmark-baseline.json --max-regression 10 \
  --json benchmark-current.json

# Measure quantiles, classification, entropy, and draws from a stored posterior.
uv run python benchmarks/run_benchmarks.py \
  --suite posterior --persons 2000 --items 30 --repeats 5 --warmups 1
```

Reports include workload sizes, backend details, runtime versions, every timing
sample, distribution summaries, and comparison status. Baselines with different
item counts, person counts, or effective backends are rejected instead of producing
misleading ratios. A detected regression returns a nonzero process status for CI use.
The hosted performance job also publishes its structured scoring report as a retained
build artifact, making individual revisions directly comparable.

## API Stability (v1.1)

Starting with v1.0, this package follows [semantic versioning](https://semver.org/).
The current release is **1.1.0**.

### Stable Public API

The following are guaranteed stable and will not have breaking changes in v1.x releases:

- **Core functions**: `fit_mirt()`, `fscores()`, `simdata()`, `itemfit()`, `personfit()`, `dif()`
- **Result classes**: `FitResult`, `ScoreResult`, `CVResult`, `BatchFitResult`
- **Model classes**: All IRT models (`TwoParameterLogistic`, `GradedResponseModel`, etc.)
- **CAT**: `CATEngine`, `CATResult`, `CATState`
- **Diagnostics**: `compare_models()`, `anova_irt()`, `compute_fit_indices()`, `sibtest()`
- **Utilities**: `bootstrap_se()`, `bootstrap_ci()`, `generate_plausible_values()`, `cross_validate()`, `fit_models()`
- **Data functions**: `load_dataset()`, `list_datasets()`, `set_dataframe_backend()`, `get_dataframe_backend()`
- **Backend selection**: `set_backend()`, `get_backend()`, `get_backend_info()`, `should_use_rust()`, `is_rust_available()`

### Experimental (may change in minor releases)

- Internal `_rust_backend` / `backends.rust` module functions (use public wrappers instead)
- MCMC samplers (`GibbsSampler`, `MHRMEstimator`) - API may be refined
- Cognitive Diagnostic Models (`DINA`, `DINO`, `fit_cdm()`) - under active development

### Versioning Policy

- **Major version (2.0, 3.0)**: Breaking API changes
- **Minor version (1.1, 1.2)**: New features, backward compatible
- **Patch version (1.0.1, 1.0.2)**: Bug fixes only

## License

MIT License - see [LICENSE](LICENSE)

## Citation

If you use this package in your research, please cite:

```bibtex
@software{mirt_python,
  author = {Lyons, Cameron},
  title = {mirt: Multidimensional Item Response Theory for Python},
  url = {https://github.com/Cameron-Lyons/mirt},
  version = {1.1.0}
}
```

## References

- Chalmers, R. P. (2012). mirt: A Multidimensional Item Response Theory Package for the R Environment. *Journal of Statistical Software*, 48(6), 1-29.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46(4), 443-459.
- Cho, A. E., Wang, C., Zhang, X., & Xu, G. (2021). Gaussian variational estimation for multidimensional item response theory. *British Journal of Mathematical and Statistical Psychology*, 74, 52-85.
- Rockova, V., & George, E. I. (2018). The spike-and-slab LASSO. *Journal of the American Statistical Association*, 113(521), 431-444.
- de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika*, 76(2), 179-199.
