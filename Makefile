.PHONY: lint fmt test test-rust test-slow bench docs develop

lint:
	uv run ruff check src tests benchmarks
	uv run ruff format --check src tests benchmarks
	uv run mypy src/mirt --ignore-missing-imports

fmt:
	uv run ruff format src tests benchmarks
	cargo fmt --all

test:
	uv run pytest

test-slow:
	uv run pytest -m slow

test-rust:
	cargo test --locked --all-features

bench:
	uv run python benchmarks/run_benchmarks.py

docs:
	cd docs && uv run sphinx-build -W --keep-going -b html . _build/html

develop:
	uv run maturin develop --release
