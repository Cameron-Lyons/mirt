# Changelog

All notable changes to the mirt package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-08-18

### Added
- Documented Rust fallback contract (`numpy` / `optional` / `required` / `mixed`)
- `should_use_rust()` honors global `set_backend("numpy")` alongside per-call flags
- Rust↔Python numerical parity tests for likelihood, E-step, scoring, and diagnostics
- User guides for CAT, DIF, multigroup, and equating; curated `examples/` scripts
- Benchmark harness under `benchmarks/` for EM fit, scoring, and CAT
- Expanded `@pytest.mark.slow` coverage for MCMC / high-cost paths

### Changed
- Modular Rust wrappers under `mirt.backends.rust` ( `_rust_backend` remains a shim)
- Coverage gate raised from 55% toward Production/Stable expectations (now 59%)
- Stricter mypy checking on additional public modules

### Included from the original 1.1 development milestone

### Added
- Custom exception hierarchy for better error handling
- CI performance smoke-test job and weekly scheduled slow-test job
- Performance regression smoke tests for import time and small-model fit time
- Documentation regression tests to prevent stale API symbol references in docs
- Improved docstring coverage across modules

### Changed
- Minimum Python version lowered from 3.14 to 3.11 for broader compatibility
- Development status updated to Production/Stable
- CI now tests on Python 3.11, 3.12, 3.13, and 3.14
- Top-level API now lazy-loads heavy equating/multigroup/plotting/report symbols
- Dataset constants in `mirt.utils.datasets` are lazily materialized on first access
- Replaced broad `except Exception` handlers with narrower recoverable exception classes
- Sphinx quickstart and API reference examples now match current public symbols
- Integration/scoring tests use backend-tolerant convergence and correlation expectations
- Reduced dead/duplicate implementation noise across EM, CAT/MCAT, scoring, and residuals
- Optimized EM dichotomous item updates with analytic gradients (`1PL`/`2PL`/`3PL`/`4PL`)
- Optimized `fit_ising` interaction-gradient computation

## [1.0.0] - 2026-01-15

### Added
- Stable public API policy for v1.x (see README API Stability)
- Computerized Adaptive Testing (CAT) module with:
  - Multiple item selection strategies (MFI, MEI, KL, Urry, random, a-stratified)
  - Configurable stopping rules (SE threshold, max/min items)
  - Exposure control methods (Sympson-Hetter, randomesque)
  - Content balancing constraints
  - CAT simulation and batch evaluation functions
- DIF analysis (LR, Wald, Lord, Raju), DTF/DRF, SIBTEST, GRDIF
- Multigroup IRT with invariance testing; bifactor models
- Zero-inflated, unfolding, testlet, CDM, and mixture IRT models
- GVEM and sparse Bayesian estimation paths
- HTML report generation, vertical scaling, fixed-item calibration / equating

### Changed
- Declared Production/Stable for core public API
- Refactored Rust backend into modular structure
- Consolidated duplicate code patterns with shared utility functions

## [0.1.11] - 2025-01-08

### Added
- Computerized Adaptive Testing (CAT) module with:
  - Multiple item selection strategies (MFI, MEI, KL, Urry, random, a-stratified)
  - Configurable stopping rules (SE threshold, max/min items)
  - Exposure control methods (Sympson-Hetter, randomesque)
  - Content balancing constraints
  - CAT simulation and batch evaluation functions

### Changed
- Refactored Rust backend into modular structure
- Consolidated duplicate code patterns with shared utility functions

## [0.1.10] - 2025-01-07

### Added
- Zero-inflated IRT models (ZI-2PL, ZI-3PL, Hurdle IRT)
- Unfolding models (GGUM, Ideal Point, Hyperbolic Cosine)
- Testlet response models for local dependence
- Cognitive Diagnostic Models (DINA, DINO)
- Mixture IRT models for latent class analysis

## [0.1.9] - 2025-01-07

### Added
- DIF (Differential Item Functioning) analysis with multiple methods:
  - Likelihood ratio test
  - Wald test
  - Lord's chi-square
  - Raju's area measures
- DTF (Differential Test Functioning) and DRF analysis
- SIBTEST implementation

## [0.1.8] - 2025-01-06

### Added
- Multigroup IRT analysis with invariance testing:
  - Configural, metric, scalar, and strict invariance
  - Likelihood ratio tests for invariance constraints
- Bifactor model support

## [0.1.7] - 2025-01-05

### Added
- MCMC estimation methods (Metropolis-Hastings, Gibbs sampling)
- Mixed effects IRT models
- LLTM (Linear Logistic Test Model)

## [0.1.6] - 2025-01-04

### Added
- Model fit statistics (M2, RMSEA, CFI, TLI, SRMSR)
- Item fit indices (S-X2, infit, outfit)
- Person fit statistics

## [0.1.5] - 2025-01-03

### Added
- Bootstrap standard errors and confidence intervals
- Plausible values generation
- Multiple imputation for missing data

## [0.1.4] - 2025-01-02

### Added
- Polytomous IRT models:
  - Graded Response Model (GRM)
  - Generalized Partial Credit Model (GPCM)
  - Partial Credit Model (PCM)
  - Nominal Response Model (NRM)
- Multidimensional IRT models (exploratory and confirmatory)

## [0.1.3] - 2025-01-01

### Added
- High-performance Rust backend via PyO3
- Parallel E-step computation with Rayon
- EAPsum scoring with Lord-Wingersky algorithm

## [0.1.2] - 2024-12-31

### Added
- EM algorithm estimation
- Multiple scoring methods (EAP, MAP, ML, WLE)
- Standard error estimation

## [0.1.1] - 2024-12-30

### Added
- Core dichotomous models (1PL, 2PL, 3PL, 4PL)
- Basic parameter estimation
- Response simulation

## [0.1.0] - 2024-12-29

### Added
- Initial release
- Basic IRT model infrastructure
- NumPy/SciPy integration
