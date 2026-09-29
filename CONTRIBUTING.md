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
uv run --no-sync ruff format src tests benchmarks
uv run --no-sync ruff check src tests benchmarks

uv run mypy src/mirt --ignore-missing-imports

uv run pytest

uv run pytest -m slow
uv run pytest -m performance
```

Convenience targets: `make lint`, `make fmt`, `make test`, `make test-slow`, `make bench`, `make develop`.

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

CI installs dependencies from `uv.lock` and builds one release wheel with
`Cargo.lock` enforced. The same ABI3 wheel is tested on Python 3.11–3.14 and used
for all performance checks and the slow suite. Native jobs explicitly require
the extension to load. A separate job runs the full non-slow, non-performance
suite from source without the extension. Workflow syntax, lint, and type checks
run before wheel builds and tests; lint and type checking do not build the
project. Test environments include plotting support.

Every native matrix job enforces a 90% coverage floor. Installed-wheel paths
are mapped back to `src/mirt` so coverage artifacts and pull-request annotations
refer to repository files. Pytest rejects unknown configuration options and
unregistered markers. Use `uv lock` when changing dependencies and commit the
resulting lockfile. CI uses uv 0.12.7.

To reproduce wheel testing in a clean environment:

```bash
uv sync --locked --no-install-project --extra dev --extra plot
uv run --no-sync maturin build --release --locked --out dist
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
Pull-request updates cancel superseded runs, and jobs have explicit time limits.
Test reports and scoring/posterior benchmarks are retained as workflow artifacts.

Workflow syntax is checked by a pinned, checksum-verified actionlint release.
Run `actionlint` to validate workflow changes locally.

## Documentation

```bash
uv pip install -e ".[docs]"
make docs
```

User guides live under `docs/guides/`; runnable scripts under `examples/`. Timing harness: `make bench`.

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

## Experimental APIs

See README “API Stability”. Experimental surfaces (for example CDM helpers and some MCMC APIs) may change in minor releases. Prefer public wrappers over private `_rust_backend` symbols.

## Pull requests

- Keep changes focused; match existing style and naming.
- Add or extend tests for behavior changes; keep smoke tests small and fast.
- Run `make lint` and the relevant `pytest` / `cargo test` suites before opening a PR.
- Do not commit generated artifacts (for example `item_analysis_report.html`, `docs/_build/`).
- Use clear commit messages that explain *why*.
