# mirt

**Multidimensional Item Response Theory for Python**

A comprehensive Python implementation of Item Response Theory (IRT) models with a high-performance Rust backend, inspired by R's [mirt](https://github.com/philchalmers/mirt) package.

## Features

### Core IRT Models
- **Dichotomous**: 1PL (Rasch), 2PL, 3PL, 4PL
- **Polytomous**: GRM, GPCM, PCM, NRM
- **Multidimensional**: Exploratory and confirmatory MIRT
- **Bifactor**: Bifactor and hierarchical models

### Advanced Models
- **Cognitive Diagnostic**: DINA, DINO, G-DINA
- **Testlet**: Random effects for item bundles
- **Nested Logit**: Keyed multiple-choice items with informative distractors
- **Mixture IRT**: Latent class IRT models
- **Zero-Inflated**: ZI-2PL, ZI-3PL, Hurdle IRT
- **Unfolding**: GGUM, Ideal Point, Hyperbolic Cosine
- **Nonparametric**: Monotonic spline IRFs
- **Network Psychometrics**: Ising and sparse Gaussian graphical models

### Estimation Methods
- **EM Algorithm**: Gauss-Hermite quadrature (with Rust acceleration)
- **GVEM**: Gaussian Variational EM for fast high-dimensional estimation
- **Sparse Bayesian**: Spike-slab LASSO for automatic structure discovery
- **MHRM**: Metropolis-Hastings Robbins-Monro
- **MCMC**: Gibbs sampling for Bayesian estimation
- **MCEM/QMCEM**: Monte Carlo EM for high dimensions

### Computerized Adaptive Testing (CAT)
- Item selection: MFI, MEI, KL divergence, a-stratified, Urry
- Stopping rules: SE threshold, max items, classification
- Exposure control: Sympson-Hetter, randomesque, progressive
- Content balancing: Blueprint constraints
- **MCAT**: Multidimensional CAT with D-optimality and trace criteria

### Diagnostics & DIF
- **Item fit**: Infit, outfit, S-X2
- **Person fit**: Zh, lz, infit/outfit
- **Model fit**: M2, RMSEA, CFI, TLI, SRMSR
- **DIF analysis**: Likelihood ratio, Wald, Lord, Raju
- **GRDIF**: Generalized Residual DIF for multiple groups with robust scaling (MAD/IQR)
- **DTF/DRF**: Differential test/response functioning
- **SIBTEST**: Simultaneous item bias test
- **Local dependence**: Q3, chi-square residuals

### Additional Features
- Custom dichotomous, ordinal, multidimensional, and latent group models
- Multiple group analysis with invariance testing
- Bootstrap standard errors and confidence intervals
- Plausible values for population inference
- Missing data imputation
- Built-in sample datasets
- Plotting (ICC, information, Wright maps, DIF)
- DataFrame output (pandas or polars)
- Fixed-item calibration and test equating
- **Vertical scaling**: Grade-level linking with growth constraints
- Reliable Change Index (RCI) for clinical significance
- Profile-likelihood confidence intervals
- Posterior parameter sampling
- **Result objects**: Validated uncertainty, confidence intervals, and portable exports
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
```

### Fitting Models

```python
result_1pl = mirt.fit_mirt(responses, model="1PL")
result_2pl = mirt.fit_mirt(responses, model="2PL")
result_3pl = mirt.fit_mirt(responses, model="3PL")

result_grm = mirt.fit_mirt(likert_data, model="GRM", n_categories=5)
result_gpcm = mirt.fit_mirt(likert_data, model="GPCM", n_categories=5)

result_mirt = mirt.fit_mirt(responses, model="2PL", n_factors=2)
```

### Person Scoring

```python
eap = mirt.fscores(result, responses, method="EAP")
map_scores = mirt.fscores(result, responses, method="MAP")
ml = mirt.fscores(result, responses, method="ML")

print(eap.theta)
print(eap.standard_error)
```

### Diagnostics

```python
item_fit = mirt.itemfit(result, responses)
print(item_fit)

person_fit = mirt.personfit(result, responses)
aberrant = person_fit[person_fit["Zh"] < -2]

fit_indices = mirt.compute_fit_indices(result.model, responses)
print(fit_indices)

results = [result_1pl, result_2pl, result_3pl]
comparison = mirt.compare_models(results)
```

### DIF Analysis

```python
groups = np.array([0] * 250 + [1] * 250)

dif_lr = mirt.dif(responses, groups, method="likelihood_ratio")

dif_wald = mirt.dif(responses, groups, method="wald")
dif_lord = mirt.dif(responses, groups, method="lord")
dif_raju = mirt.dif(responses, groups, method="raju")

from mirt.diagnostics.dif import compute_grdif

groups_multi = np.array(["A"] * 200 + ["B"] * 200 + ["C"] * 200)
grdif_result = compute_grdif(
    responses, groups_multi,
    model="2PL",
    scaling_method="mad",
)
print(f"Flagged items: {np.where(grdif_result['flagged_rs'])[0]}")
```

### Multiple Group Analysis

```python
from mirt.multigroup import fit_multigroup, compare_invariance

result = fit_multigroup(responses, groups, model="2PL", invariance="metric")

results = compare_invariance(responses, groups, model="2PL", verbose=True)
```

### Computerized Adaptive Testing

```python
from mirt.cat import CATEngine

cat = CATEngine(result.model, se_threshold=0.3, max_items=20)

sim_results = cat.run_batch_simulation(
    true_thetas=np.linspace(-2, 2, 11),
    n_replications=100,
)

state = cat.get_current_state()
while not state.is_complete:
    item = state.next_item
    response = get_examinee_response(item)
    state = cat.administer_item(response)

final = cat.get_result()
print(final.summary())

from mirt.cat import MCATEngine

mcat = MCATEngine(
    mirt_model,
    selection_method="D-optimality",
    max_items=30,
)
mcat_result = mcat.run_simulation(true_theta=np.array([0.5, -0.3]))
print(f"Estimated theta: {mcat_result.theta}")
print(f"Covariance: {mcat_result.theta_cov}")
```

### Advanced Models

```python
from mirt import fit_cdm
q_matrix = np.array([[1, 0], [1, 1], [0, 1], [1, 1]])
cdm_result = fit_cdm(responses, q_matrix, model="DINA")

from mirt import fit_mixture_irt
mix_model, class_posteriors = fit_mixture_irt(
    responses, n_classes=2, base_model="2PL"
)

from mirt import TestletModel, create_testlet_structure
testlet_struct = create_testlet_structure(n_items=20, testlet_sizes=[5, 5, 5, 5])
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
result = estimator.fit(model, responses)

print(f"Effective dimensions: {result.effective_dimensionality}")
print(f"Sparsity ratio: {1 - result.sparsity_pattern.mean():.1%}")
print(result.loading_table())

estimator = GVEMEstimator(max_iter=200, tol=1e-4)
result = estimator.fit(model, responses)
```

### Test Equating & Calibration

```python
from mirt.utils import fixed_calib, equate, Q3, residuals

calib_result = fixed_calib(
    responses=combined_responses,
    anchor_model=existing_model,
    anchor_items=[0, 1, 2, 3, 4],
)
print(f"New item difficulties: {calib_result.new_difficulty}")

equating = equate(
    model_old=form_a_model,
    model_new=form_b_model,
    anchor_items_old=[0, 1, 2],
    anchor_items_new=[0, 1, 2],
    method="stocking_lord",
)
print(f"Scale transformation: theta_new = {equating.A:.3f} * theta_old + {equating.B:.3f}")

q3_matrix = Q3(result.model, responses, scores.theta)
resid = residuals(result.model, responses, scores.theta)
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

result = vertical_scale(
    grade_data,
    method="chain",
    enforce_monotonicity=True,
)

print(f"Grade means: {result.grade_means}")
print(f"Growth curve: {result.growth_curve}")

diagnostics = compute_vertical_diagnostics(result, grade_data)
print(f"Grade separation (effect sizes): {diagnostics.grade_separation}")
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
plot_category_curves(result.model, item_idx=0)

plot_person_item_map(result.model, scores.theta)
```

## Supported Models

### Dichotomous Models

| Model | Description | Parameters |
|-------|-------------|------------|
| 1PL/Rasch | One-parameter logistic | difficulty (b) |
| 2PL | Two-parameter logistic | discrimination (a), difficulty (b) |
| 3PL | Three-parameter logistic | a, b, guessing (c) |
| 4PL | Four-parameter logistic | a, b, c, upper asymptote (d) |

### Polytomous Models

| Model | Description | Use Case |
|-------|-------------|----------|
| GRM | Graded Response Model | Ordered categories (Likert) |
| GPCM | Generalized Partial Credit | Partial credit scoring |
| PCM | Partial Credit Model | Rasch for polytomous |
| NRM | Nominal Response Model | Unordered categories |
| 2PL/3PL/4PL-NRM | Nested Logit | Keyed multiple choice with distractor information |

### Advanced Models

| Model | Description |
|-------|-------------|
| MIRT | Multidimensional IRT |
| Bifactor | General + specific factors |
| DINA/DINO | Cognitive diagnostic |
| Testlet | Local dependence modeling |
| Nested Logit | Keyed response and conditional distractor modeling |
| Mixture IRT | Latent class IRT |
| GGUM | Generalized graded unfolding |

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `fit_mirt()` | Fit IRT models |
| `fscores()` | Person ability estimation |
| `simdata()` | Simulate response data |
| `itemfit()` | Item fit statistics |
| `personfit()` | Person fit statistics |
| `dif()` | DIF analysis |
| `load_dataset()` | Load sample datasets |

### Estimator Classes

| Class | Description |
|-------|-------------|
| `EMEstimator` | Standard EM with Gauss-Hermite quadrature |
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
| `vertical_scale()` | Vertical scaling for grade linking |
| `mirt.diagnostics.psis_loo()` | PSIS-LOO with optional threaded observation smoothing |

### Utility Functions

| Function | Description |
|----------|-------------|
| `bootstrap_se()` | Bootstrap standard errors with optional process workers |
| `bootstrap_ci()` | Percentile, basic, and BCa intervals with parallel replicate and jackknife fits |
| `parametric_bootstrap()` | Deterministic model-based bootstrap with optional process workers |
| `generate_plausible_values()` | Plausible values |
| `combine_plausible_values()` | Validated scalar or vector combining with Rubin uncertainty |
| `plausible_value_statistics()` | Mean, variance, SD, or percentiles for one or all latent factors |
| `plausible_value_regression()` | Validated ordinary or weighted regression with combined uncertainty |
| `cross_validate()` | Validated K-fold, stratified, group-aware, and leave-one-out evaluation with optional process parallelism |
| `impute_responses()` | Missing data imputation |
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
| `fixed_calib()` | Fixed-item calibration for test equating |
| `equate()` | Test form equating (Stocking-Lord, Haebara, mean/sigma) |
| `RCI()` | Reliable Change Index for clinical significance |
| `PLCI()` | Profile-likelihood confidence intervals |
| `draw_parameters()` | Draw samples from posterior distribution |
| `posterior_summary()` | Summarize sampled parameter uncertainty |
| `sample_expected_scores()` | Propagate parameter uncertainty to expected scores |
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

## Comparison with R mirt

| Feature | R mirt | Python mirt |
|---------|--------|-------------|
| Dichotomous models | 1PL-4PL | 1PL-4PL |
| Polytomous models | GRM, GPCM, PCM, NRM | GRM, GPCM, PCM, NRM |
| Multidimensional | Full support | Full support |
| Bifactor | Yes | Yes |
| Cognitive diagnostic | mirtCAT separate | Built-in (DINA, DINO) |
| Estimation | EM, MHRM, MCMC | EM, GVEM, Sparse Bayesian, MHRM, MCMC |
| Automatic structure discovery | No | Yes (spike-slab LASSO) |
| CAT | mirtCAT package | Built-in (unidimensional + MCAT) |
| DIF | Yes | Yes (LR, Wald, Lord, Raju, GRDIF) |
| Multiple groups | Full support | Full support |
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
| **Calibration** | Fixed-item calibration EM algorithm, Stocking-Lord equating criterion |
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
| **pandas** | DataFrame output for results | `pip install mirt[pandas]` |
| **polars** | DataFrame output (faster, preferred when both installed) | `pip install mirt[polars]` |

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
