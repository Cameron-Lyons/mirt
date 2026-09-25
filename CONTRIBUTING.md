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

## Experimental APIs

See README “API Stability”. Experimental surfaces (for example CDM helpers and some MCMC APIs) may change in minor releases. Prefer public wrappers over private `_rust_backend` symbols.

## Pull requests

- Keep changes focused; match existing style and naming.
- Add or extend tests for behavior changes; keep smoke tests small and fast.
- Run `make lint` and the relevant `pytest` / `cargo test` suites before opening a PR.
- Do not commit generated artifacts (for example `item_analysis_report.html`, `docs/_build/`).
- Use clear commit messages that explain *why*.
