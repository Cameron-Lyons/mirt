# Contributing to mirt

Thanks for contributing! This guide covers local setup, checks, and pull request expectations.

## Setup

Requires Python 3.11+, Rust (stable), and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Cameron-Lyons/mirt.git
cd mirt
uv sync --locked --no-install-project --extra dev --extra plot
uv run --no-sync maturin develop --release --locked --uv
```

Optional extras: `.[docs]`, `.[plot]`, `.[pandas]`, `.[polars]`, `.[gpu]`.

## Python checks

```bash
uv run --no-sync ruff format src tests benchmarks .github/scripts
uv run --no-sync ruff check src tests benchmarks .github/scripts

uv run --no-sync mypy src/mirt --ignore-missing-imports

uv run --no-sync pytest

uv run --no-sync pytest -m slow
uv run --no-sync pytest -m performance
```

Convenience targets: `make lint`, `make fmt`, `make test`, `make test-slow`, `make test-performance`, `make bench`, `make develop`.

## Rust backend contract

Wrappers live under `src/mirt/backends/rust/`. Each module declares `FALLBACK_MODE`:

- `numpy` — pure NumPy path when Rust is missing or `mirt.set_backend("numpy")`
- `optional` — returns `None`; callers keep a Python implementation
- `required` — raises; use the public Python estimator instead
- `mixed` — more than one mode in the same module

Prefer public APIs and `mirt.should_use_rust()` over private `_rust_backend` symbols.

## Rust checks

The extension lives under `rust_src/` with the workspace `Cargo.toml` at the repo root.

```bash
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-targets --all-features
```

Or: `make test-rust` / `make fmt`.

## CI checks

CI installs dependencies from `uv.lock`, packages a source distribution, and
builds one release wheel from its extracted contents with `Cargo.lock` enforced.
This checks that the source distribution contains everything needed to build
and run the package. The same ABI3 wheel is tested on Python 3.11–3.14 and used
for all performance checks and the slow suite. Native jobs explicitly require
the extension to load. A separate job runs the full non-slow, non-performance
suite from source without the extension. Workflow syntax, lint, and type checks
run before wheel builds and tests; lint and type checking do not build the
project. Core test environments include plotting support.

A required CPU PyTorch job installs the locked `dev` and `gpu` extras on
macOS ARM64, whose PyTorch wheel does not require the Linux CUDA libraries.
It runs all tensor likelihood, posterior, GVEM, and fit parity tests against
NumPy references. The job requires a working CPU tensor runtime; only the
explicit CUDA hardware test skips, and its reason appears in the test report.
To run those tests locally after installing the GPU extra:

```bash
uv run --no-sync pytest tests/test_gpu -m 'not slow and not performance' -rs
```

Every native matrix job enforces a 90% coverage floor. Installed-wheel paths
are mapped back to `src/mirt` so coverage artifacts and pull-request annotations
refer to repository files. Pytest rejects unknown configuration options and
unregistered markers. Use `uv lock` when changing dependencies and commit the
resulting lockfile. CI uses uv 0.12.7.

To reproduce wheel testing in a clean environment:

```bash
uv sync --locked --no-install-project --extra dev --extra plot
uv run --no-sync maturin sdist --out dist
sdist_dir=$(mktemp -d)
tar -xzf dist/*.tar.gz -C "$sdist_dir" --strip-components=1
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv run --directory "$sdist_dir" --no-sync \
  maturin build --release --locked --out "$PWD/dist"
uv pip install --no-deps --no-index dist/*.whl
uv run --no-sync pytest -m 'not slow and not performance' --cov=mirt
```

Use `--no-sync` after installing the wheel so the test command keeps that exact
installation. To reproduce the NumPy-only job, use a clean checkout with no
built extension, install dependencies with
`uv sync --locked --no-install-project --extra dev --extra plot`, then run:

```bash
PYTHONPATH=src uv run --no-sync pytest -m 'not slow and not performance'
```

Slow tests run weekly and can also be enabled through the CI workflow's manual
`run-slow` input. `CI complete` fails if a required job fails, is cancelled, or
is unexpectedly skipped; it can be selected as a required branch-protection
check. The existing `Rust checks (Cargo.toml)` check covers formatting, Clippy,
and Rust tests separately; documentation and security retain separate checks.
Documentation builds run for pull requests, merge queues, and manual dispatches.
Pull-request updates cancel superseded runs, and jobs have explicit time limits.
Test reports and scoring, posterior, equating, and pointwise-likelihood benchmarks
are retained as workflow artifacts.

Workflow syntax is checked by a pinned, checksum-verified actionlint release.
Run `actionlint` to validate workflow changes locally.

Release builds also enforce `Cargo.lock`. The full non-slow, non-performance
suite runs against every published wheel architecture: Linux x86_64 and ARM64,
Windows x64, and macOS Intel and Apple Silicon. Linux x86_64 wheels additionally
run on every supported Python version. The uploaded source distribution is
rebuilt and tested separately, and publication requires both wheel and source
distribution validation to pass. Release test reports are retained for review.
An additional prerequisite checks the actual metadata inside every wheel and
source distribution. Package names and versions must agree with the release
tag, or with `Cargo.toml` for an untagged build. Missing source distributions,
stale artifacts, ambiguous metadata, and unexpected publish inputs fail before
the PyPI publication environment is entered. The checker normalizes PEP 440
pre, post, and dev releases, including Cargo spellings such as `v1.2.3-rc.1`,
while requiring three release components and rejecting PyPI local versions.
To check collected artifacts with the locked development dependencies:

```bash
uv run --no-sync python .github/scripts/check_release_artifacts.py dist --tag v1.2.3
```

Automatic version-update PRs synchronize the runtime version, Cargo package
metadata, and the root Cargo lock entry. Their helper validates all versions
before editing and preserves an unversioned dynamic project entry in `uv.lock`.

Python security checks export every runtime and optional dependency from
`uv.lock` to a temporary PEP 751 lockfile and audit it with pinned pip-audit
2.10.1. This covers versions selected for other supported Python versions and
platforms, without installing the optional GPU stack. The audit fails on known
vulnerabilities or collection failures and retains its input and JSON findings.

## Documentation

```bash
uv pip install -e ".[docs]"
make docs
```

User guides live under `docs/guides/`; runnable scripts under `examples/`. Timing harness: `make bench`.

The `multigroup-fit` suite times ten EM updates for three simulated populations
of `--persons` respondents each. It covers metric constraints with distinct
group item contexts, pooled scalar constraints, and fixed anchors. Each run
starts from a fresh model and records timing and peak traced allocations.
CI records this suite with the scoring and posterior-likelihood benchmarks.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/run_benchmarks.py \
  --suite multigroup-fit --persons 2000 --items 30 --repeats 3 --warmups 1 \
  --backend rust --json multigroup-fit.json
```

The `binary-likelihood` suite measures public single-person and batched
likelihoods across ten binary model families with 10% missing responses.
It covers matched ability points, one shared ability point, and grids of
21 and 121 points. Inputs and parameters are prepared outside timing, and
each workload records a separate peak of Python/NumPy allocations:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/run_benchmarks.py \
  --suite binary-likelihood --persons 5000 --items 60 --repeats 7 --warmups 2 \
  --backend numpy --json binary-likelihood.json
```

The optional `gpu-likelihood` suite requires PyTorch and times real tensor
likelihoods for 1PL, 2PL, 3PL, multidimensional 2PL, GRM, GPCM, and PCM, plus a
complete 2PL E-step. Category cases include items with different category
counts and 10% missing responses. It uses CUDA when available and otherwise
CPU tensors; JSON reports include the tensor device and PyTorch version, and
baseline comparisons require matching runtime metadata. It reports time only
because Python allocation tracing does not measure tensor memory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/run_benchmarks.py \
  --suite gpu-likelihood --persons 2000 --items 40 --repeats 7 --warmups 2 \
  --json tensor-likelihood.json
```

Prepared item and joint likelihoods, exact uncertainty, native fitting, and
response compression require original built-in model hooks. A shared check
compares instance attributes, class methods, and inherited definitions captured
when models are defined, including changes made before estimator imports.
Custom curves, setters, layouts, and joint likelihoods retain their public
evaluation paths. Conditional item objectives only require unchanged item
hooks; a joint likelihood override alone does not disable them. Polytomous
single-person and batched likelihoods both follow public probability methods.
Binary likelihoods also preserve original ability points and custom response
validation, with constant curves broadcast across the requested grid.
These checks preserve lazy model imports and keep default numerical shortcuts
available.

The `qmcem-mstep` suite measures one complete QMCEM M-step on a precomputed
256-point shared grid for 2PL, three-factor 2PL/MIRT, four-category GRM, and
two-factor GPCM/NRM models with 10% missing responses:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite qmcem-mstep --persons 1000 --items 10 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-qmcem-mstep.json
```

Timing includes expected-count accumulation, item optimization, and copying the
template model; sampling is excluded. Traced allocations exclude the response
matrix, shared grid, and posterior weights created before measurement. Logistic
and category objectives reuse the EM analytic gradients. Custom model curves
evaluate only the shared grid during numerical optimization. Independent sample
grids and custom estimator item callbacks retain the Monte Carlo update path.
Importance normalization centers likelihoods before exponentiation and consumes
one owned copy, preserving cached likelihood arrays and unit posterior mass for
large common offsets. Grid, posterior, and count outputs still grow with their
respective person/sample/item dimensions.

The `qmcem-fit` suite uses the same six models and input generation to measure
likelihood refresh, E-steps including likelihood reporting, optional item
uncertainty, and complete two-iteration fits:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite qmcem-fit --persons 1000 --items 10 --repeats 3 --warmups 1 --backend numpy --json /tmp/mirt-qmcem-fit.json
```

Refresh timing includes the owned likelihood output; fits include validation,
sampling, likelihood refreshes, expected counts, optimization, and model copying.
Shared samples use the public batch likelihood directly. Independent samples and
custom sample callbacks retain the Monte Carlo evaluation path. Likelihood and
posterior matrices still grow with the person and sample counts.

The `regularized` suite measures E-steps and four-iteration fits with three
coordinate-descent sweeps per M-step. It covers two factors with 15 quadrature
points per factor and three factors with nine points, on complete responses
and responses with 10% missing values:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite regularized --persons 5000 --items 50 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-regularized.json
```

Use `--backend rust` to exercise native coordinate descent. E-step timing
excludes input creation, while complete fits include validation, preparation,
and latent-density updates. Traced allocations include outputs and cached
response components. Posterior normalization reuses its likelihood buffer;
only the returned person-by-quadrature array grows with both dimensions.

To compare response-pattern grouping on repeated and mostly distinct rows:

```bash
uv run --no-sync python benchmarks/run_benchmarks.py --suite patterns --persons 100000 --items 20 --repeats 7 --backend numpy
uv run --no-sync python benchmarks/run_benchmarks.py --suite patterns --persons 100000 --items 20 --repeats 7 --backend rust
```

Scoring and `collapse_patterns` share this grouping implementation. The Rust
path hashes borrowed integer rows and releases the GIL during grouping; the
NumPy fallback sorts compact row keys. Both preserve first-appearance order,
missing-code normalization, and full-width integer response codes.

The `posterior` suite measures posterior summaries and highest-density intervals
separately. Both use a precomputed two-factor posterior so fitting is excluded:

```bash
uv run --no-sync python benchmarks/run_benchmarks.py --suite posterior --persons 20000 --items 20 --repeats 7 --backend rust
```

Use `--backend numpy` to compare the fallback. Posterior sampling uses NumPy's
seeded random stream on both backends; only the independent CDF searches move to
Rust, so draws are identical across backends and batching choices.

The `bayesian` suite measures WAIC using 1,000 posterior draws over person × item
observations, and PSIS-LOO using 4,000 draws over person-level observations with
normal and heavy-tailed log likelihoods. Inputs are precomputed, and PSIS runs
serially with per-observation relative efficiencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 uv run --no-sync python benchmarks/run_benchmarks.py --suite bayesian --persons 1000 --items 20 --repeats 9 --warmups 2 --backend numpy --json /tmp/mirt-bayesian.json
```

Predictive summaries validate and reduce bounded observation blocks, with at
least one full posterior column per block. They reuse owned exponential buffers
and center variance calculations before reduction. PSIS uses a partition for its
tail cutoff and sorts only the selected tail; parallel calls bound queued work
by worker count. Reports trace Python/NumPy allocations, excluding input creation
and native allocations. Input dtype conversion and returned observation arrays
can still grow with input size.

The `data` suite covers pairwise availability counts, mode imputation, and item
statistics on five-category responses with 10% missing data:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite data --persons 20000 --items 200 --repeats 7 --warmups 2 --backend numpy
```

Keep BLAS thread settings the same when comparing runs. Pairwise counts use
bounded floating-point matrix products with exact integer accumulation;
imputation and item statistics share bounded category counting.

The `diagnostics` suite measures shared NumPy Q3 correlations and LD chi-square/G²
statistics on complete data and data with 15% missing entries. Timing excludes
model fitting and probability evaluation. Each workload also records a separate
`tracemalloc` peak for Python/NumPy allocations, rather than process RSS:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite diagnostics --persons 20000 --items 150 --repeats 7 --warmups 1 --backend numpy --json /tmp/mirt-diagnostics.json
```

Residual utilities, local-dependence diagnostics, testlet Q3, and the native
backend's NumPy fallback share the correlation kernel. It processes bounded row
blocks and skips missing-data matrix products for complete blocks. Each caller
retains its minimum pair count, undefined-value, and diagonal conventions.

LD statistics and the native backend's NumPy fallback also share a bounded-row
kernel. It reuses transposed cross-classification cells and derives observed
cells from margins in complete blocks. Expected cells use direct probability
products to preserve accuracy near zero and one; expected-count floors and
pair eligibility are applied after all blocks have accumulated.

The `misfit` benchmark suite measures `identify_misfitting_patterns` on complete
and incomplete 2PL/GRM data. It includes probability evaluation, fit totals,
and returned flag dictionaries, while excluding simulation and ability scoring:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 uv run --no-sync python benchmarks/run_benchmarks.py --suite misfit --persons 20000 --items 100 --repeats 5 --warmups 1 --backend numpy --json /tmp/mirt-misfit.json
```

Misfit identification evaluates standardized residuals in row blocks sized by
item/category count. It retains per-person/item sums and flagged entries instead
of full residual matrices or response-pattern summaries. The returned lists still
grow with the number of flagged entries. Models without batch metadata retain
their itemwise probability fallback. Traced peaks measure Python/NumPy allocations,
not process RSS or native allocations.

To measure native likelihoods, GRM/GPCM/PCM fits, MAP/ML scoring, and repetitive
EM data, use the `kernels` and `optimization` suites:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 uv run --no-sync python benchmarks/run_benchmarks.py --suite kernels --suite optimization --persons 1000 --items 20 --repeats 5 --warmups 1 --backend rust --json /tmp/mirt-optimization.json
OPENBLAS_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite information --persons 500 --items 20 --repeats 3 --backend numpy --json /tmp/mirt-information.json
```

The `information` suite measures the marginal information matrix and records a
separate `tracemalloc` peak for Python/NumPy allocations. This is not process RSS
or a measurement of native allocations. Its computational cost remains quadratic
in the number of free parameters, while retained person arrays are bounded.
Reports include thread environment settings and reject comparisons with different
settings when the baseline records them.

The `latent-density` suite measures Gaussian parameter updates and log-density
evaluation in one, three, and eight dimensions. `--persons` sets the number of
input points; input generation is excluded from timing and traced memory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 uv run --no-sync python benchmarks/run_benchmarks.py --suite latent-density --persons 100000 --items 20 --repeats 9 --warmups 2 --backend numpy --json /tmp/mirt-latent-density.json
```

Updates form weighted covariance matrices directly, using scratch storage linear
in the point count and dimension count instead of one covariance matrix per
point. Log-density evaluation reuses the owned point buffer for centering and
reduces quadratic forms without a separate elementwise-product matrix. Both
preserve caller arrays. The reported `tracemalloc` peaks cover Python/NumPy
allocations and exclude process RSS and native library workspace.

Fast paths use exact built-in model types; custom and multidimensional scoring
or polytomous optimization retain their generic implementations. Native M-steps
use analytic gradients and a projected BFGS optimizer with backtracking, so
fixed-iteration fits can differ slightly from SciPy fits. Compare the objective,
parameter recovery, and convergence as well as speed. Itemwise EM standard errors
and full marginal-information standard errors retain their separate objectives.

Core EM E-steps return per-person **log** marginal likelihoods internally; avoid
converting them to probabilities for convergence or fit statistics. Prepared fit
contexts retain compressed rows and their frequencies through standard errors,
cache bounded response components for generic expected-count matrix products,
and own any Python/Rust worker pools. Contexts release the pools on success and
on exceptions. The legacy `mirt._rust_backend` module remains a lazy compatibility
namespace; new internal code should import from the owning backend module.

The `fit-statistics` benchmark suite measures item/person mean squares on
complete and incomplete data, plus full 2PL/GRM item- and person-fit diagnostics
with population-wide p-value adjustment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 uv run --no-sync python benchmarks/run_benchmarks.py --suite fit-statistics --persons 20000 --items 150 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-fit-statistics.json
```

Mean-square reductions use bounded row blocks, reuse residual storage, and
apply denominator thresholds after accumulation. Item- and person-fit diagnostics
share one probability evaluation per block across all requested statistics; block
sizes account for the number of response categories. Small inputs still use
one probability call. Reports record timing and a separate `tracemalloc` peak
for Python/NumPy allocations, including probability evaluation for item/person fit.
This measures temporary allocation, not process RSS or native allocation.
Item-fit reductions reuse the mean-square accumulator, and direct `compute_s_x2`
calls share the same streaming path. S-X2 score groups are calculated once from
all respondents; group-count eligibility and p-value adjustments are applied
after accumulation.

The `model-fit` benchmark suite measures full M2/RMSEA/CFI/TLI/SRMSR diagnostics
for 2PL and GRM models with complete and incomplete data, using both supplied
abilities and quadrature integration. Timing includes probability evaluation;
fitting and data generation are excluded:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 uv run --no-sync python benchmarks/run_benchmarks.py --suite model-fit --persons 20000 --items 150 --repeats 5 --warmups 1 --backend numpy --json /tmp/mirt-model-fit.json
```

Model-fit calculations stream response and probability blocks, accounting for
category width, and retain item-pair matrices rather than respondent-wide moment
arrays. Empirical expected moments share observed pair counts; the independence
baseline reuses observed pairwise means. Missing pairs and undefined correlations
retain their existing conventions. Traced peaks measure Python/NumPy temporary
allocations, not process RSS or native allocations; the quadrature grid itself
still grows with the number of latent dimensions.

The `weighted-em` suite measures survey-weighted E-steps for 2PL, two-factor
2PL, and five-category GRM models, plus five-iteration 2PL/GRM fits including
standard errors. Simulated responses contain 10% missing data; survey weights
range from 0.25 to 2.0. The two-factor E-step uses seven points per dimension, and the
unidimensional workloads use 21 points. Input generation is excluded:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 uv run --no-sync python benchmarks/run_benchmarks.py --suite weighted-em --persons 1000 --items 30 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-weighted.json
```

Built-in models evaluate probabilities once per quadrature grid; custom models
retain per-person likelihood calls. E-steps return log marginals internally so
convergence and fit statistics remain meaningful for long tests. Traced memory
covers Python/NumPy allocations, excluding native workspace and process RSS.
Weighted standard errors share the core EM item-curvature implementation,
including analytic built-in derivatives and numerical custom-model fallbacks.

The `weighted-mstep` suite isolates item optimization and standard errors for
2PL, three-factor 2PL, GRM, and NRM on a fixed posterior. Responses contain 10%
missing data, and every eleventh survey weight is zero. It uses 21 quadrature
points for unidimensional models and seven per dimension for three-factor 2PL.
Input preparation is excluded; copying the model for each optimization is
included. Weighted counts use bounded response blocks, without constructing a
dense survey-weighted posterior:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite weighted-mstep --persons 5000 --items 20 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-weighted-mstep.json
```

The `polytomous-fit` suite measures M-steps and five-iteration ordinary and
survey-weighted fits for GRM, GPCM, PCM, and NRM. GRM/GPCM/NRM include one and
two factors; items cycle through two to five categories, with 10% missing
responses. Unidimensional grids use 15 points and two-factor grids use seven
per dimension. The suite forces the Python M-step for ordinary EM; supported
unidimensional models otherwise retain their native optimizer. Ordinary fits
omit standard errors, while weighted fits include them:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite polytomous-fit --persons 1000 --items 8 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-polytomous-fit.json
```

Input generation is excluded. Timing includes copying the model for M-steps;
complete fits include validation and response preparation. Built-in category
objectives compute likelihoods and gradients together without trial model
updates. Custom probability methods and parameter layouts use numerical
optimization. Traced memory covers Python/NumPy allocations.

The `item-curvature` suite measures standalone central, forward, and Richardson
uncertainty with fixed posterior inputs. It covers unidimensional and
three-factor 2PL, GPCM, and two-factor GRM/NRM, with 10% missing responses.
Category counts cycle through two to five. Central differences also run with
two workers; quadrature uses 21 points for one factor and seven per dimension
otherwise. Counts are prepared once per call and shared across parameter fields
and Richardson steps. Parallel workers retain custom model state and instance
methods in one isolated model per item:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite item-curvature --persons 5000 --items 20 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-item-curvature.json
```

These methods compute diagonal complete-data curvature. The benchmark step is
`1e-4`, and outputs preserve parameter shapes with zeros for fixed coordinates.
Traced memory includes count preparation and worker models, while excluding
input creation and native allocations. Posterior arrays remain read-only.

The `bl-fit` suite measures five-iteration joint BL optimization and complete
fits, including diagonal marginal standard errors. It covers 2PL and 3PL,
two-factor 2PL/MIRT/NRM, unidimensional GRM/GPCM, and three-factor bifactor
models. Category counts cycle through two to four, with 10% missing responses.
Quadrature uses 21 points for one factor and five per dimension otherwise:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite bl-fit --persons 1000 --items 8 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-bl-fit.json
```

Input creation is excluded; timing includes model copying, validation, and
response preparation. Built-in joint gradients reuse the clipped item kernels
and aggregate counts from the current posterior. Trial parameters remain in an
isolated model, response preparation is reused for curvature, and row blocks
bound likelihood scratch space. The person-by-quadrature output still scales
with sample and grid size. Traced memory covers Python/NumPy allocations.
Custom models, likelihoods, and derivative-free methods use their numerical
objective.

The `irtree-fit` suite measures E-steps, node-count aggregation, expected
complete-data uncertainty, and complete fits with up to five EM iterations.
It covers the Bockenholt, extreme-midpoint, and direction-intensity trees with
five response categories, 10% missing responses, and seven quadrature points
per trait:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite irtree-fit --persons 1000 --items 8 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-irtree-fit.json
```

Stage timing excludes input creation and uses fixed posterior inputs for counts
and uncertainty. Complete fits include model copying, pseudo-item expansion,
response preparation, item and trait-distribution updates, scores, and standard
errors. Small response components are cached within a fit; row and node blocks
bound larger scratch arrays. Posterior normalization consumes owned likelihood
storage and copies borrowed custom buffers. Previous posteriors are released
before the next E-step. The posterior and count outputs still scale with sample,
node, and grid sizes. Memory peaks measure Python/NumPy allocations, excluding
input creation and native workspace. Uncertainty retains each node's expected
complete-data 2×2 information convention and leaves unvisited or singular nodes
undefined.

The `mcem-fit` suite measures person-specific likelihood refresh, an E-step
including likelihood reporting, an M-step, optional item uncertainty, and a
complete two-iteration MCEM fit for three-factor 2PL/MIRT, 3PL, and two-factor
GRM/GPCM/NRM models. Each person has 64 samples and 10% of responses
are missing:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite mcem-fit --persons 600 --items 6 --repeats 5 --warmups 1 --backend numpy --json /tmp/mirt-mcem-fit.json
```

Refresh and M-step inputs are fixed across runs. M-steps and fits include model
copying; complete fits include parameter initialization, sampling, likelihood
updates, and item optimization. Reported E-steps reset the random generator and
use precomputed prior inputs. Ordinary MCEM, QMCEM, and stochastic fits reuse
fresh E-step importance normalizers or accepted posterior likelihoods for their
current likelihood report. They release previous draws and weights before the
next E-step, and refresh the retained final draw after the last item update.
Custom sampling, likelihood, normalization, and prior hooks retain separate
evaluation. Built-in item objectives prepare small observed blocks
once and stream bounded blocks for larger draws. Custom models and estimator
item objectives use the numerical probability path. Category probabilities
retain MCEM's upper clip at one; binary probabilities clip at `1-epsilon`.
Input creation is excluded from timing and memory tracing. Person-specific
sample and weight inputs still scale with persons, samples, and factors, and
the likelihood output grows with persons and samples. Ordinary item likelihoods
reduce bounded point blocks against unexpanded responses. Float32 and strided
sample inputs are converted only in those blocks. Public probability callbacks
retain their behavior and borrowed buffers; custom likelihood and validation
methods use the existing Monte Carlo model path. Traced peaks cover Python/NumPy
allocations and exclude native workspace. This suite measures importance
sampling MCEM.

Uncertainty stages enable `compute_standard_errors=True` on fixed draws and
weights. Built-in polytomous items use exact diagonal curvature; logistic and
affine items differentiate prepared analytic gradients. Both use bounded
person-specific blocks or shared-grid expected counts. Polytomous curvature
leaves the model untouched; gradient trials restore it after each evaluation.
Custom item objectives use numerical likelihood curvature with their
observed-person inputs. These are diagonal complete-data
approximations with draws and weights held fixed; they exclude missing
information, covariance between parameters, and sampling uncertainty. Fixed
coordinates have zero errors; unobserved items and undefined curvature retain
NaN errors. `se_step_size` controls finite differences, and does not affect exact
polytomous curvature. Ordinary fits retain the default disabled mode and
placeholder errors.

The `mcem-sampling` suite covers posterior MCEM with 64 draws per person and
stochastic EM with five chains, using three-factor 2PL, two-factor GRM, and
three-factor 2PL/six-factor MIRT with correlated priors and nonzero means.
It also measures opt-in uncertainty on precomputed Gaussian draws with uniform
weights. Responses have 10% missing values:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite mcem-sampling --persons 600 --items 6 --repeats 5 --warmups 1 --backend numpy --json /tmp/mirt-mcem-sampling.json
```

Prior-kernel timing excludes input generation. Gaussian kernels stream owned
point blocks, using diagonal scaling or triangular solves, and rescale only
overflowing reductions to preserve representable tail kernels and tiny terms.
E-step timing includes the initial prior draw and 20 Metropolis transitions;
complete fits include two iterations, model copying, and all sampling and item
updates. Each repetition resets the random generator to the same seed. Input
generation is excluded. Proposal construction reuses its owned transform
buffer; temporary normals are released before likelihood/prior evaluation.
Accepted cells copy directly into chain state. Initial likelihood and prior
outputs are copied to protect borrowed custom callback buffers. Proposed
likelihoods are saved in reusable storage before calling the prior, allowing
callbacks to share scratch. Acceptance combines likelihood and prior differences
separately, preserving prior ratios under large common likelihood offsets.
Draws, proposals, and likelihood outputs
still grow with persons, samples, and factors; Gaussian reduction scratch is
bounded independently of draw count.

The `variational` suite measures the shared NumPy E-step used by GVEM and sparse
Bayesian estimation in one, three, and six dimensions. Each call starts from the
same local bound values and performs three inner iterations with 10% missing
responses. It calls the NumPy paths directly; input generation is excluded:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite variational --persons 10000 --items 100 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-variational.json
```

Row blocks are sized by item and factor counts, with at least one respondent per
block. Means, covariance matrices, and local variational bounds still scale with
sample size. The reported peaks measure Python/NumPy allocations during E-steps;
they exclude input creation, native workspace, process RSS, and other fitting
stages. The benchmark does not measure complete-fit speedups.

The `gvem-uncertainty` suite measures diagonal standard errors on a precomputed
variational state and complete five-iteration 2PL GVEM fits, including standard
errors, in one and three dimensions. Simulated responses contain 10% missing
data. GPU dispatch is disabled; `--backend` selects the other fitting kernels:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 uv run --no-sync python benchmarks/run_benchmarks.py --suite gvem-uncertainty --persons 1000 --items 20 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-gvem-uncertainty.json
```

At the default SE step size, built-in 1PL/2PL models use exact diagonal curvature
of the same bound with the variational distribution held fixed. Custom objectives,
estimator/model subclasses, and nondefault step sizes retain numerical
differentiation. This preserves the diagonal curvature convention; it does not
compute a full covariance matrix or profile the variational distribution.
Curvature scratch space is bounded by row and factor counts, and missing items
retain undefined standard errors. Traced peaks exclude input generation, native
workspace, and process RSS.

The `variational-objective` suite measures the shared NumPy logistic bound and
Gaussian prior/entropy calculation for GVEM and sparse Bayesian estimation in
one, three, and six dimensions. Inputs include correlated covariance matrices,
shifted priors, and 10% missing responses. Variational state generation and fitting
are excluded; both callers use NumPy regardless of the selected backend:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite variational-objective --persons 10000 --items 100 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-variational-objective.json
```

The objective processes respondent blocks sized by item and factor counts, with
at least one whole respondent per block. Missing observations are excluded from
the logistic bound, while every respondent contributes a Gaussian KL term. The
sparse loading prior remains separate. Traced peaks cover Python/NumPy scratch
allocations, excluding input generation, native workspace, and process RSS.

The `variational-mstep` suite measures NumPy GVEM and sparse Bayesian item updates
for 2PL models in one, three, and six dimensions, plus fixed-loading 1PL updates.
It includes statistic accumulation, parameter solves, and resetting item state
between repeats. Posterior state generation and complete fitting are excluded:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite variational-mstep --persons 10000 --items 100 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-variational-mstep.json
```

Both estimators accumulate curvature, scores, and weighted means in bounded
respondent blocks, retaining item-level statistics. Intercepts reuse these
statistics after loadings are updated. Fixed-loading models omit covariance
calculations; GVEM retains its itemwise least-squares fallback for singular
curvature. These workloads exercise NumPy on either backend selection. Traced
peaks exclude input generation, native workspace, and process RSS.

The ``adaptive-scoring`` suite measures EAP updates for a fixed history of up to
15 administered items from a larger bank. It covers 1D and 2D 2PL plus 2D and 3D
MIRT. Grids use 21 points in one dimension and 11 per factor otherwise. Timing
includes curve evaluation and full covariance reduction; input creation, item
selection, exposure/content control, and simulation are excluded. Repeats reuse
the parameter-independent quadrature grid. Traced memory covers Python/NumPy
allocations and excludes native workspace and process RSS:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python benchmarks/run_benchmarks.py --suite adaptive-scoring --items 200 --repeats 7 --warmups 2 --backend numpy --json /tmp/mirt-adaptive-scoring.json
```

## Experimental APIs

See README “API Stability”. Experimental surfaces (for example CDM helpers and some MCMC APIs) may change in minor releases. Prefer public wrappers over private `_rust_backend` symbols.

## Pull requests

- Keep changes focused; match existing style and naming.
- Add or extend tests for behavior changes; keep smoke tests small and fast.
- Run `make lint` and the relevant `pytest` / `cargo test` suites before opening a PR.
- Do not commit generated artifacts (for example `item_analysis_report.html`, `docs/_build/`).
- Use clear commit messages that explain *why*.
