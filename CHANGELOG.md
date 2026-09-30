# Changelog

All notable changes to the mirt package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Reused IRTree response preparation across E/M-steps and uncertainty, bounded
  node/row likelihood scratch space, and normalized owned likelihood buffers
  in place. Released previous posteriors before the next E-step and reused the
  final posterior on convergence. Batched bounded node information matrices,
  avoiding unused correct-count calculations during uncertainty. Shared the
  clipped logistic objective with ordinary item fitting. Added an `irtree-fit`
  timing and traced-memory benchmark suite.
- Reused clipped item-gradient kernels for joint BL marginal optimization of
  built-in 1PL–4PL, GRM, GPCM, PCM, NRM, MIRT, and bifactor models. Shared
  response preparation with diagonal marginal curvature, bounded likelihood
  scratch space, and isolated trial model state. Added a `bl-fit` timing and
  traced-memory benchmark suite. Custom likelihoods retain numerical optimization.
- Shared numerical item curvature across EM and standalone standard errors.
  Prepared bounded posterior counts once per uncertainty call, reused them
  across Richardson step sizes, and used one worker pool for all item fields.
  Fixed and padded coordinates avoid unnecessary probability trials. Added an
  `item-curvature` timing and traced-memory benchmark suite.
- Prepared analytic GRM, GPCM, PCM, and NRM item objectives for Python EM and
  survey-weighted EM, including multidimensional models. Category counts use
  bounded accumulation, and trial evaluation leaves model parameters intact.
  Added a `polytomous-fit` timing and traced-memory benchmark suite.
- Accumulated survey-weighted binary and category counts in bounded blocks,
  eliminating dense weighted posterior copies from optimization and standard
  errors. Weighted EM now shares core item objectives and analytic gradients;
  fixed items and items with no weighted observations retain their parameters.
  Failed numerical trials restore parameter state. Added a `weighted-mstep`
  timing and traced-memory benchmark suite.
- Reused response preparation across regularized E/M-steps, bounded response
  and probability scratch space, and removed unused item masks. Shared in-place
  posterior normalization across ordinary, survey-weighted, and regularized EM.
  Added a `regularized` timing and traced-memory benchmark suite.
- Prepared Python 1PL–4PL item objectives once per optimization and consolidated
  their likelihood and gradient calculations. Added a `logistic-fit` benchmark
  suite covering unidimensional models and multidimensional 2PL.
- Prepared analytic M-step objectives for MIRT and bifactor models, sharing
  quadrature design arrays and expected counts without mutating trial model
  parameters. Added a `multidimensional-fit` benchmark suite including
  confirmatory constraints and missing responses.
- Shared bounded full, single-item, and paired probability evaluation for MIRT
  and bifactor models. Reused owned logit buffers on the ordinary logistic path
  and preserved sparse bifactor loading access. Added a
  `multidimensional-probability` timing and memory benchmark suite.
- Shared stable logistic variances and bounded multidimensional information
  reductions. Test matrices contract only active factor pairs, with a fast path
  for single-person item matrices used by adaptive selection. Added a
  `multidimensional-information` timing and memory benchmark suite.
- Shared bounded full, single-item, and paired probability evaluation across
  1PL–4PL models. Added a `logistic-probability` benchmark suite covering all
  query forms and multidimensional 2PL models.
- Shared bounded query handling across ULL, CLL, and NLL, and reused the symmetric
  sigmoid derivative in unipolar curves and logistic information. Added a
  `unipolar` benchmark suite for probability, information, and paired queries.
- Shared bounded information evaluation across 1PL–4PL models, reusing buffers
  and keeping multidimensional slope norms factored during evaluation. Added a
  `logistic-information` benchmark suite for full and single-item queries.
- Shared CLL/NLL probability and information kernels with bounded evaluation
  and in-place buffers. Extended the `asymmetric` benchmark suite to both links.
- Shared bounded 5PL probability and information evaluation with in-place
  buffers, restricting log-domain reductions and overflow recovery to affected
  cells. Added an `asymmetric` benchmark suite for full and paired probabilities
  and information.
- Shared ability validation and bounded test-information reductions between
  information and reliability utilities. Streamed item-information and expected
  scores, evaluated sparse item selections directly, and reused repeated item
  queries. Added a `curves` timing and traced-memory benchmark suite.
- Bounded information evaluation for reliability and measurement-error summaries
  by cohort blocks. Added a `reliability` benchmark suite with binary and ordinal
  models, including timing and traced-memory measurements.
- Shared binary item and total-score moments in classical statistics, using
  bounded, centered item-total correlation blocks and prepared NumPy deletion
  statistics. Removed redundant response copies and reused variance buffers.
  Added a `classical` benchmark suite for both correlation modes and missing data.
- Streamed empirical RMSEA probability evaluation and binned residual reductions,
  removing the dense person-by-bin matrix and unused bin-mean calculations.
  Sparse histograms retain only occupied bins within each block. Added an
  `empirical` benchmark suite for binary and polytomous fit with 10 or 100 bins.
- Calibrated kernel-smoothed item curves in bounded grid blocks with in-place
  Gaussian buffers, restricting missing-item fallbacks to affected grid points.
  Added a `kernel-smoothing` benchmark suite for weighted complete/missing data
  on standard and dense grids, with timing and traced-memory measurements.
- Shared Gaussian smoothing between calibration and empirical `itemGAM` curves,
  reusing kernels across items and bounding temporary memory for both means and
  uncertainty bands. Extended the smoothing benchmarks to binary and polytomous
  empirical curves with complete and missing responses.
- Batched built-in survey-weighted EM likelihoods over quadrature points and
  retained log marginals through convergence and fit statistics, avoiding
  likelihood underflow on long tests. Reused posterior normalization storage
  and shared item-curvature calculations for weighted standard errors,
  preserving fixed-parameter masks and avoiding repeated finite differences.
  Added a `weighted-em` timing and traced-memory benchmark suite.
- Shared Gaussian variational E-step calculations between GVEM's NumPy
  fallback and sparse Bayesian estimation. Respondent blocks bound temporary
  arrays while preserving missing-data bounds, shifted priors, and each
  estimator's lower bound on local variational parameters. Small samples use
  one block without copying outputs. Added a `variational` timing and memory suite.
- Shared bounded M-step statistics between NumPy GVEM and sparse Bayesian fitting,
  reusing weighted means for intercept updates and batching GVEM item solves.
  Fixed loadings skip covariance calculations. Added a `variational-mstep`
  benchmark suite for free- and fixed-loading updates and traced allocations.
- Shared the NumPy variational objective between GVEM and sparse Bayesian
  estimation, bounding likelihood and Gaussian KL scratch space by respondent
  blocks and reusing temporary arrays. Added a `variational-objective` benchmark
  suite for timing and traced allocations on precomputed variational states.
- Computed built-in GVEM standard errors directly from diagonal variational
  curvature at the default step size, avoiding full-objective finite differences
  and cancellation against parameter-independent terms. Curvature reductions
  use bounded row blocks. Custom objectives and nondefault steps retain a shared
  numerical fallback that restores parameters on failed trial evaluations.
  Added a `gvem-uncertainty` benchmark suite for uncertainty and complete fits.
- Streamed model-fit moments in bounded response and probability blocks, sharing
  pairwise counts and reusing observed moments for the independence baseline.
  Complete blocks avoid missing-data matrix products. Added a `model-fit`
  benchmark suite for empirical/quadrature integration and traced memory peaks.
- Shared bounded-memory LD chi-square/G² calculations between local-dependence
  diagnostics and the NumPy backend fallback, reusing transposed cells and
  complete-block observed margins to reduce matrix products. Extended the
  `diagnostics` benchmark suite with complete/missing-data LD workloads.
- Streamed misfit identification in bounded probability blocks, retaining only
  flagged responses and fit totals instead of a full residual analysis and
  unused response-pattern summaries. Added complete/missing-data 2PL and GRM
  timing and traced-memory workloads to the `misfit` benchmark suite.
- Consolidated pairwise residual correlations across diagnostics, testlet models,
  and NumPy backend fallbacks into one bounded-memory kernel. Complete-data blocks
  use a single matrix product, and column offsets improve numerical stability.
- Added a `diagnostics` benchmark suite for complete/missing-data Q3 timing and
  traced Python/NumPy memory peaks.
- Shared bounded WAIC/PSIS predictive-density reductions, reusing exponential
  buffers and centering variance calculations for large log-likelihood shifts.
  PSIS partitions its tail cutoff, sorts the selected tail once, and bounds
  queued thread tasks. Added Bayesian diagnostic timing and memory workloads.
- Kept EM convergence and final likelihoods in log space to avoid underflow on
  long tests, and reused built-in likelihood buffers for posterior normalization.
- Batched generic EM expected counts with bounded matrix products and cached
  response preparation; reused Python and Rust worker pools within each fit.
- Retained response-pattern compression through native 2PL standard errors,
  weighting posterior contributions by pattern frequencies.
- Made the legacy Rust-backend namespace lazy and routed internal imports to
  their owning backend modules.
- Cached bounded item/category probability tables in native likelihood and E-step
  kernels, with contiguous output storage, in-place posterior normalization,
  zero-copy posterior transfers, and GIL release during native fitting.
- Added batched Rust GRM/GPCM/PCM M-steps with analytic gradients and bounded
  optimization, and native unidimensional 1PL–4PL MAP/ML scoring.
- Compressed repetitive EM data into frequency-weighted response patterns while
  preserving sample counts, latent-density updates, and uncertainty objectives.
- Streamed marginal-information perturbations to remove quadratic person-array
  caching; built-in unidimensional EM models use analytic item curvature at the
  default SE step setting to avoid finite-difference cancellation.
- Added likelihood, optimization, and information benchmark suites, including
  traced information-memory peaks and BLAS/Rayon thread metadata.
- Replaced per-point Gaussian covariance tensors with weighted matrix products
  and reused owned point buffers during updates and log-density evaluation.
  Added a `latent-density` benchmark suite for timing and traced memory.
- Accelerated pairwise availability counts with bounded BLAS matrix products
  and exact integer accumulation.
- Shared memory-bounded category counting between imputation and item statistics.
- Added a `data` benchmark suite for counts, imputation, and item statistics.
- Shared response-pattern grouping between ability scoring and collapse utilities,
  with Rust hashing over compact integer rows and GIL release during grouping.
- Reduced copies and sorting work in the NumPy pattern-grouping fallback while
  preserving first-appearance order and full-width response codes.
- Added a `patterns` benchmark suite for repeated and mostly distinct responses.
- Shared exact row-wise CDF searches across posterior quantiles, intervals, and
  sampling, with Rust acceleration and batched reproducible draws.
- Moved shortest-interval selection to a linear Rust scan with a reusable grid buffer.
- Bounded highest-density interval working memory by the full joint grid and
  removed redundant marginal reductions when all coordinates are unique.

### Fixed
- Preserved IRTree posterior normalization for large common log-likelihood
  offsets and protected borrowed custom likelihood buffers. Explicit node masks
  suppress stored decisions consistently in likelihoods and counts. IRTree
  optimizer gradients now respect clipped tails. Custom theta overrides retain
  numerical EM item objectives for logistic and affine models.
- Restored BL parameters after failed numerical optimization or curvature
  trials, and skipped optimization for models with no free parameters.
- Preserved custom model constructor state and bound instance probability
  methods in parallel numerical uncertainty. Core EM uncertainty falls back
  to numerical curvature for custom curve and parameter overrides.
- Enabled item updates and access for category-by-factor parameter arrays,
  fixing multidimensional NRM fitting. Shared coordinate-wise numerical
  curvature now preserves tensor shapes in EM and standalone standard errors,
  with parameter restoration when probability evaluation fails.
- Kept regularized MIRT convergence, likelihoods, and fit statistics in log space
  so long tests do not underflow into a likelihood floor or falsely converge.
  Posterior weights preserve unit row sums and prior differences under large
  common log-likelihood offsets. Failed adaptive warmstarts restore the penalty
  and adaptive weights, and response preparation is released after each fit.
- Matched Python 1PL–4PL EM gradients to the clipped likelihood, preserved small
  positive-tail derivatives, and avoided NaNs from zero counts at saturated
  probabilities. Custom binary models now retain their own probability functions
  during optimization. Multidimensional 2PL objectives share stable logit recovery.
- Preserved confirmatory loading patterns across model copies and excluded
  fixed zero loadings from optimization and parameter counts. Parallel EM
  workers now isolate numerical trial updates, including custom model state.
  Bifactor copies retain subclass behavior. Affine M-step gradients follow
  probability clipping and handle zero counts at saturated probabilities.
- Recovered finite MIRT and bifactor logits after product overflow, intercept
  cancellation, or cancellation between factor contributions. Probabilities
  and scalar/matrix information now share this recovery, while unrelated
  bifactor columns and genuinely nonfinite inputs retain their prior semantics.
- Retained multidimensional and bifactor information in both logistic tails,
  recovering representable scalar and matrix entries after intermediate slope
  overflow or tail underflow. Standardized MIRT loadings and communalities now
  retain large finite slopes without overflowing their squared norms.
- Logistic probabilities recover finite results from overflowing offsets, and
  4PL curves retain exact asymptotes. Multidimensional 2PL probability and
  information recenter large locations before subtraction, with exact arithmetic
  limited to cells whose severe cancellation survives centered evaluation.
- Unipolar probability and information preserve both tails and small nonzero
  information near the curve peak, including extreme slope rescaling. Corrected
  the docstring to describe the implemented symmetric curve with a 0.25 peak.
- Logistic Fisher information preserves both tails when success probabilities
  round to one, and recovers representable results after extreme slope scaling.
  Multidimensional 2PL information avoids premature overflow in slope norms and
  location offsets; NaN abilities propagate instead of appearing uninformative.
- CLL/NLL information preserves representable tails when probabilities round to
  zero or one, including recovery after slope rescaling. Extreme finite offsets
  no longer overflow before small slopes can rescale them, and probabilities
  retain their exact limits at infinite abilities.
- Strongly asymmetric 5PL curves preserve probabilities and Fisher information
  when the underlying sigmoid rounds to zero or one. Generalized difficulty
  inversion retains interior solutions at extreme asymmetries and targets near
  probability boundaries instead of incorrectly clamping to the search limits.
  Finite abilities and parameters also retain representable results when their
  initial differences or logit products overflow before asymmetry rescaling.
- Information and expected-score queries accept NumPy integer indices and item
  selection arrays, preserve repeated selection order, and interpret a single
  multidimensional ability vector consistently. Invalid indices and integration
  inputs now fail validation before model evaluation.
- Empirical reliability preserves score-unit invariance for tiny and very large
  scores and standard errors. Marginal reliability normalizes large or tiny
  density weights safely, handles normal-density tails, and retains positive
  variances without an arbitrary minimum. Invalid item information is rejected
  before summing it into test information.
- Classical item-total correlations remain accurate and within [-1, 1] on long,
  nearly constant tests, including exactly constant corrected totals.
- Empirical quantile bins handle extreme finite abilities without overflowing
  their edges or means. Empirical RMSEA accepts shared category counts from
  custom ordinal models and rejects malformed category metadata.
- Kernel calibration preserves person weights when observations share very large
  distances, excludes zero-weight neighbors from extreme-distance centering, and
  keeps ordinary grid points independent of overflowing-distance fallbacks.
- `itemGAM` produces finite curves with tiny positive bandwidths and preserves
  uncertainty in grid tails when nearby responses are missing.
- Highest-density intervals no longer depend on row position or batch size at
  adjacent floating-point probability boundaries, or return reversed bounds at
  very small positive probability levels.
- Benchmark comparisons reject different person counts for pattern grouping
  and the new highest-density interval workload.

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
