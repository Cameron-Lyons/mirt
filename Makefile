.PHONY: lint fmt test test-rust docs develop

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src/mirt --ignore-missing-imports

fmt:
	uv run ruff format src tests
	cargo fmt --all

test:
	uv run pytest

test-rust:
	cargo test --all-features

docs:
	cd docs && uv run sphinx-build -b html . _build/html

develop:
	uv run maturin develop --release
