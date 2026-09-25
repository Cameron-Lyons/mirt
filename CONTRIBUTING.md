# Contributing to mirt

Thanks for contributing! This guide covers local setup, checks, and pull request expectations.

## Setup

Requires Python 3.11+, Rust (stable), and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Cameron-Lyons/mirt.git
cd mirt
uv sync --locked --extra dev --extra plot --no-install-project
uv run --no-sync maturin develop --release --locked --uv
```

Optional extras: `.[docs]`, `.[plot]`, `.[pandas]`, `.[polars]`, `.[gpu]`.

## Python checks

```bash
uv run ruff format src tests benchmarks
uv run ruff check src tests benchmarks

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

## Documentation

```bash
uv pip install -e ".[docs]"
make docs
```

User guides live under `docs/guides/`; runnable scripts under `examples/`. Timing harness: `make bench`.

## Continuous integration

The main CI workflow validates workflow syntax, lint, and types before running
the Python 3.11–3.14 native test matrix, a separate NumPy-only test environment,
and all tests marked `performance`. Test environments include plotting support,
and every native test job enforces a 90% coverage floor. Pytest rejects unknown
configuration options and unregistered markers.
Python dependencies come from `uv.lock`; native builds and Rust checks require
`Cargo.lock` to remain unchanged. Use `uv lock` when changing dependencies and
commit the resulting lockfile. CI uses uv 0.12.7.

`CI checks` aggregates these jobs and fails on failures, cancellations, or
unexpected skips. It can be selected as a required branch-protection check;
Rust, documentation, and security workflows retain their separate checks.
The slow suite runs weekly and can also be enabled through the CI workflow's
manual `run-slow` input. Pull-request updates cancel superseded runs, and jobs
have explicit time limits. Test reports and scoring/posterior benchmarks are
retained as workflow artifacts.

Run `actionlint` to validate workflow changes locally. To reproduce the
NumPy-only job, use a clean checkout with no built extension, install dependencies
with `uv sync --locked --extra dev --extra plot --no-install-project`, then run:

```bash
PYTHONPATH=src uv run --no-sync pytest -m 'not slow and not performance'
```

## Experimental APIs

See README “API Stability”. Experimental surfaces (for example CDM helpers and some MCMC APIs) may change in minor releases. Prefer public wrappers over private `_rust_backend` symbols.

## Pull requests

- Keep changes focused; match existing style and naming.
- Add or extend tests for behavior changes; keep smoke tests small and fast.
- Run `make lint` and the relevant `pytest` / `cargo test` suites before opening a PR.
- Do not commit generated artifacts (for example `item_analysis_report.html`, `docs/_build/`).
- Use clear commit messages that explain *why*.
