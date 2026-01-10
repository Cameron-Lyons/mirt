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
- **Mixture IRT**: Latent class IRT models
- **Zero-Inflated**: ZI-2PL, ZI-3PL, Hurdle IRT
- **Unfolding**: GGUM, Ideal Point, Hyperbolic Cosine
- **Nonparametric**: Monotonic spline IRFs

### Estimation Methods
- **EM Algorithm**: Gauss-Hermite quadrature (with Rust acceleration)
- **MHRM**: Metropolis-Hastings Robbins-Monro
- **MCMC**: Gibbs sampling for Bayesian estimation
- **MCEM/QMCEM**: Monte Carlo EM for high dimensions

### Computerized Adaptive Testing (CAT)
- Item selection: MFI, MEI, KL divergence, a-stratified, Urry
- Stopping rules: SE threshold, max items, classification
- Exposure control: Sympson-Hetter, randomesque, progressive
- Content balancing: Blueprint constraints

### Diagnostics & DIF
- **Item fit**: Infit, outfit, S-X2
- **Person fit**: Zh, lz, infit/outfit
- **Model fit**: M2, RMSEA, CFI, TLI, SRMSR
- **DIF analysis**: Likelihood ratio, Wald, Lord, Raju
- **DTF/DRF**: Differential test/response functioning
- **SIBTEST**: Simultaneous item bias test
- **Local dependence**: Q3, chi-square residuals

### Additional Features
- Multiple group analysis with invariance testing
- Bootstrap standard errors and confidence intervals
- Plausible values for population inference
- Missing data imputation
- Built-in sample datasets
- Plotting (ICC, information, Wright maps, DIF)
- DataFrame output (pandas or polars)
- Fixed-item calibration and test equating
- Reliable Change Index (RCI) for clinical significance
- Profile-likelihood confidence intervals
- Posterior parameter sampling

## Installation

```bash
pip install mirt
```

With optional dependencies:
```bash
pip install mirt[pandas]    # DataFrame support via pandas
pip install mirt[polars]    # DataFrame support via polars
pip install mirt[dev]       # Development tools
```

For plotting support:
```bash
pip install matplotlib
```

## Quick Start

```python
import mirt

# Load a sample dataset
dataset = mirt.load_dataset("LSAT7")
responses = dataset["data"]

# Fit a 2PL model
result = mirt.fit_mirt(responses, model="2PL")
print(result.summary())

# Score respondents
scores = mirt.fscores(result, responses, method="EAP")
print(scores.to_dataframe().head())
```

## Examples

### Simulating Data

```python
import mirt
import numpy as np

# Basic simulation
responses = mirt.simdata(model="2PL", n_persons=500, n_items=20, seed=42)

# With specific parameters
a = np.random.lognormal(0, 0.3, size=20)
b = np.random.normal(0, 1, size=20)
responses = mirt.simdata(model="2PL", discrimination=a, difficulty=b, n_persons=1000)

# Polytomous data
likert_data = mirt.simdata(model="GRM", n_categories=5, n_persons=500, n_items=15)
```

### Fitting Models

```python
# Dichotomous models
result_1pl = mirt.fit_mirt(responses, model="1PL")
result_2pl = mirt.fit_mirt(responses, model="2PL")
result_3pl = mirt.fit_mirt(responses, model="3PL")

# Polytomous models
result_grm = mirt.fit_mirt(likert_data, model="GRM", n_categories=5)
result_gpcm = mirt.fit_mirt(likert_data, model="GPCM", n_categories=5)

# Multidimensional
result_mirt = mirt.fit_mirt(responses, model="2PL", n_factors=2)
```

### Person Scoring

```python
# Different methods
eap = mirt.fscores(result, responses, method="EAP")
map_scores = mirt.fscores(result, responses, method="MAP")
ml = mirt.fscores(result, responses, method="ML")

# Access estimates
print(eap.theta)            # Ability estimates
print(eap.standard_error)   # Standard errors
```

### Diagnostics

```python
# Item fit
item_fit = mirt.itemfit(result, responses)
print(item_fit)

# Person fit
person_fit = mirt.personfit(result, responses)
aberrant = person_fit[person_fit["Zh"] < -2]

# Model fit indices
fit_indices = mirt.compute_fit_indices(result.model, responses)
print(fit_indices)

# Model comparison
results = [result_1pl, result_2pl, result_3pl]
comparison = mirt.compare_models(results)
```

### DIF Analysis

```python
groups = np.array([0] * 250 + [1] * 250)

# Likelihood ratio test
dif_lr = mirt.dif(responses, groups, method="likelihood_ratio")

# Other methods
dif_wald = mirt.dif(responses, groups, method="wald")
dif_lord = mirt.dif(responses, groups, method="lord")
dif_raju = mirt.dif(responses, groups, method="raju")
```

### Multiple Group Analysis

```python
from mirt.multigroup import fit_multigroup, compare_invariance

# Fit with specific invariance level
result = fit_multigroup(responses, groups, model="2PL", invariance="metric")

# Compare all invariance levels
results = compare_invariance(responses, groups, model="2PL", verbose=True)
# Returns: {'configural': ..., 'metric': ..., 'scalar': ..., 'strict': ...}
```

### Computerized Adaptive Testing

```python
from mirt.cat import CATEngine

# Create CAT engine
cat = CATEngine(result.model, se_threshold=0.3, max_items=20)

# Run a simulation
sim_results = cat.run_batch_simulation(
    true_thetas=np.linspace(-2, 2, 11),
    n_replications=100,
)

# Interactive session
state = cat.get_current_state()
while not state.is_complete:
    item = state.next_item
    response = get_examinee_response(item)  # Your function
    state = cat.administer_item(response)

final = cat.get_result()
print(final.summary())
```

### Advanced Models

```python
# Cognitive Diagnostic Models
from mirt import fit_cdm
q_matrix = np.array([[1, 0], [1, 1], [0, 1], [1, 1]])
cdm_result = fit_cdm(responses, q_matrix, model="DINA")

# Mixture IRT
from mirt import fit_mixture_irt
mix_result = fit_mixture_irt(responses, n_classes=2, model="2PL")

# Testlet Model
from mirt import TestletModel, create_testlet_structure
testlet_struct = create_testlet_structure(n_items=20, testlet_sizes=[5, 5, 5, 5])
```

### Test Equating & Calibration

```python
from mirt.utils import fixed_calib, equate, Q3, residuals

# Fixed-item calibration: calibrate new items to existing scale
calib_result = fixed_calib(
    responses=combined_responses,
    anchor_model=existing_model,
    anchor_items=[0, 1, 2, 3, 4],  # Items with known parameters
)
print(f"New item difficulties: {calib_result.new_difficulty}")

# Test form equating
equating = equate(
    model_old=form_a_model,
    model_new=form_b_model,
    anchor_items_old=[0, 1, 2],
    anchor_items_new=[0, 1, 2],
    method="stocking_lord",  # or "haebara", "mean_sigma", "mean_mean"
)
print(f"Scale transformation: theta_new = {equating.A:.3f} * theta_old + {equating.B:.3f}")

# Local dependence analysis
q3_matrix = Q3(result.model, responses, scores.theta)
resid = residuals(result.model, responses, scores.theta)
print(f"Max Q3 (off-diagonal): {np.max(np.abs(np.triu(q3_matrix, 1))):.3f}")
```

### Plotting

```python
from mirt import plot_icc, plot_information, plot_person_item_map

# Item characteristic curves
plot_icc(result.model, item_idx=[0, 1, 2])

# Test information function
plot_information(result.model)

# Wright map
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

### Advanced Models

| Model | Description |
|-------|-------------|
| MIRT | Multidimensional IRT |
| Bifactor | General + specific factors |
| DINA/DINO | Cognitive diagnostic |
| Testlet | Local dependence modeling |
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

### Diagnostic Functions

| Function | Description |
|----------|-------------|
| `compute_fit_indices()` | M2, RMSEA, CFI, TLI |
| `compare_models()` | AIC/BIC comparison |
| `anova_irt()` | Likelihood ratio tests |
| `compute_dtf()` | Differential test functioning |
| `compute_drf()` | Differential response functioning |
| `sibtest()` | SIBTEST DIF detection |

### Utility Functions

| Function | Description |
|----------|-------------|
| `bootstrap_se()` | Bootstrap standard errors |
| `bootstrap_ci()` | Bootstrap confidence intervals |
| `generate_plausible_values()` | Plausible values |
| `impute_responses()` | Missing data imputation |
| `set_dataframe_backend()` | Choose pandas/polars |
| `residuals()` | Model residuals (raw, standardized, Pearson, deviance) |
| `Q3()` | Yen's Q3 local dependence statistic |
| `LD_X2()` | Chen & Thissen LD chi-square |
| `fixed_calib()` | Fixed-item calibration for test equating |
| `equate()` | Test form equating (Stocking-Lord, Haebara, mean/sigma) |
| `RCI()` | Reliable Change Index for clinical significance |
| `PLCI()` | Profile-likelihood confidence intervals |
| `draw_parameters()` | Draw samples from posterior distribution |
| `randef()` / `fixef()` | Random/fixed effects from mixed models |

### Data Transformation Functions

| Function | Description |
|----------|-------------|
| `key2binary()` | Score multiple choice with answer key |
| `poly2dich()` | Convert polytomous to dichotomous |
| `reverse_score()` | Reverse score items |
| `expand_table()` | Expand frequency table to response matrix |
| `collapse_table()` | Collapse responses to frequency table |
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
| Estimation | EM, MHRM, MCMC | EM, MHRM, MCMC |
| CAT | mirtCAT package | Built-in |
| DIF | Yes | Yes (LR, Wald, Lord, Raju) |
| Multiple groups | Full support | Full support |
| Rust acceleration | No | Yes (see below) |

## Rust Acceleration

When the Rust backend is available (automatically built during installation), the following operations are accelerated with parallel processing:

| Category | Accelerated Operations |
|----------|------------------------|
| **Likelihood** | Log-likelihood computation for 2PL, 3PL, and multidimensional models |
| **EM Algorithm** | E-step (posterior computation), M-step (Newton-Raphson optimization), full EM fitting |
| **Scoring** | EAP scores, WLE scores, Lord-Wingersky recursion for sum scores |
| **Diagnostics** | Q3 matrix, LD chi-square, infit/outfit statistics, standardized residuals |
| **Calibration** | Fixed-item calibration EM algorithm, Stocking-Lord equating criterion |
| **SIBTEST** | Beta statistic computation, all-items SIBTEST |
| **CAT** | Item information, item selection, EAP updates, batch simulation |
| **Simulation** | Response generation for 2PL/3PL, GRM, GPCM |
| **Bootstrap** | Index generation, resampling, parallel bootstrap fitting |
| **Plausible Values** | Posterior sampling, MCMC generation |
| **MCMC** | Gibbs sampling for 2PL, MHRM estimation |

All Rust functions fall back to pure Python implementations if the extension is not available. The Rust backend provides significant speedups for large datasets (1000+ persons) due to:

- **Rayon parallelization**: Computation across persons or items runs in parallel
- **SIMD optimizations**: Vectorized arithmetic where available
- **Memory efficiency**: Reduced allocations compared to NumPy broadcasting

To check if Rust acceleration is available:
```python
from mirt._rust_backend import RUST_AVAILABLE
print(f"Rust backend: {'enabled' if RUST_AVAILABLE else 'disabled'}")
```

## Requirements

### Core Dependencies (always required)
- Python >= 3.11
- numpy >= 1.24
- scipy >= 1.9

### Optional Dependencies

| Package | Purpose | Installation |
|---------|---------|--------------|
| **matplotlib** | Plotting (ICC, information curves, Wright maps, DIF) | `pip install matplotlib` |
| **pandas** | DataFrame output for results | `pip install mirt[pandas]` |
| **polars** | DataFrame output (faster, preferred when both installed) | `pip install mirt[polars]` |

When neither pandas nor polars is installed, functions that return DataFrames will raise an `ImportError` with installation instructions. Plotting functions similarly require matplotlib.

To set your preferred DataFrame backend explicitly:
```python
import mirt
mirt.set_dataframe_backend("pandas")  # or "polars"
```

## Development

```bash
# Clone and install
git clone https://github.com/Cameron-Lyons/mirt.git
cd mirt
pip install -e ".[dev]"

# Build Rust extension
maturin develop --release

# Run tests
pytest

# Type checking
mypy src/mirt

# Formatting
black src tests
ruff check src tests
```

## API Stability (v1.0)

Starting with v1.0, this package follows [semantic versioning](https://semver.org/).

### Stable Public API

The following are guaranteed stable and will not have breaking changes in v1.x releases:

- **Core functions**: `fit_mirt()`, `fscores()`, `simdata()`, `itemfit()`, `personfit()`, `dif()`
- **Result classes**: `FitResult`, `ScoreResult`, `CVResult`, `BatchFitResult`
- **Model classes**: All IRT models (`TwoParameterLogistic`, `GradedResponseModel`, etc.)
- **CAT**: `CATEngine`, `CATResult`, `CATState`
- **Diagnostics**: `compare_models()`, `anova_irt()`, `compute_fit_indices()`, `sibtest()`
- **Utilities**: `bootstrap_se()`, `bootstrap_ci()`, `generate_plausible_values()`, `cross_validate()`, `fit_models()`
- **Data functions**: `load_dataset()`, `list_datasets()`, `set_dataframe_backend()`

### Experimental (may change in minor releases)

- Internal `_rust_backend` module functions (use public wrappers instead)
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
  version = {1.0.2}
}
```

## References

- Chalmers, R. P. (2012). mirt: A Multidimensional Item Response Theory Package for the R Environment. *Journal of Statistical Software*, 48(6), 1-29.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46(4), 443-459.
- de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika*, 76(2), 179-199.
