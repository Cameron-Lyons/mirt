# Contributing to mirt

Thanks for contributing! This guide covers local setup, checks, and pull request expectations.

## Setup

Requires Python 3.11+, Rust (stable), and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Cameron-Lyons/mirt.git
cd mirt
uv venv
uv pip install -e ".[dev]"

# Build the Rust extension into the active environment
uv run maturin develop --release
```

Optional extras: `.[docs]`, `.[plot]`, `.[pandas]`, `.[polars]`, `.[gpu]`.

## Python checks

```bash
# Format / lint
uv run ruff format src tests
uv run ruff check src tests

# Type check
uv run mypy src/mirt --ignore-missing-imports

# Tests (skips @pytest.mark.slow by default)
uv run pytest

# Slow / performance markers
uv run pytest -m slow
uv run pytest -m performance
# or: uv run pytest tests/test_performance_smoke.py
```

Convenience targets: `make lint`, `make fmt`, `make test`, `make develop`.

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
# opens docs/_build/html
```

## Experimental APIs

See README “API Stability”. Experimental surfaces (for example CDM helpers and some MCMC APIs) may change in minor releases. Prefer public wrappers over private `_rust_backend` symbols.

## Pull requests

- Keep changes focused; match existing style and naming.
- Add or extend tests for behavior changes; keep smoke tests small and fast.
- Run `make lint` and the relevant `pytest` / `cargo test` suites before opening a PR.
- Do not commit generated artifacts (for example `item_analysis_report.html`, `docs/_build/`).
- Use clear commit messages that explain *why*.
