.PHONY: lint fmt test test-rust test-slow test-performance bench docs develop

lint:
	uv run --no-sync ruff check src tests benchmarks .github/scripts
	uv run --no-sync ruff format --check src tests benchmarks .github/scripts
	uv run --no-sync mypy src/mirt --ignore-missing-imports

fmt:
	uv run --no-sync ruff format src tests benchmarks .github/scripts
	cargo fmt --all

test:
	uv run --no-sync pytest

test-slow:
	uv run --no-sync pytest -m slow

test-performance:
	uv run --no-sync pytest -m performance

test-rust:
	cargo test --locked --all-targets --all-features

bench:
	uv run --no-sync python benchmarks/run_benchmarks.py

docs:
	cd docs && uv run --no-sync sphinx-build -W --keep-going -b html . _build/html

develop:
	uv run --no-sync maturin develop --release --locked --uv
