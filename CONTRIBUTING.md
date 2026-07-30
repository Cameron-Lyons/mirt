# Contributing to mirt

Thanks for contributing! This guide covers local setup, checks, and pull request expectations.

## Setup

Requires Python 3.11+, Rust (stable), and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Cameron-Lyons/mirt.git
cd mirt
uv venv
uv pip install -e ".[dev]"

uv run maturin develop --release
```

Optional extras: `.[docs]`, `.[plot]`, `.[pandas]`, `.[polars]`, `.[gpu]`.

## Python checks

```bash
uv run ruff format src tests
uv run ruff check src tests

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
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

Or: `make test-rust` / `make fmt`.

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
