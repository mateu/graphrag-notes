.PHONY: all build test clean run-ml run-cli setup

# Default target
all: build

# Build the Rust CLI
build:
	cargo build --release

# Build in debug mode
build-debug:
	cargo build

# Run all tests
test: test-rust

# Run Rust tests (unit tests, no ML worker needed)
test-rust:
	cargo test

# Run Rust integration tests (requires ML worker)
test-rust-integration:
	cargo test -- --ignored

# Run Python tests
test-python:
	@echo "Python worker removed; no Python tests to run."

# Clean build artifacts
clean:
	cargo clean
	rm -rf python/.venv python/__pycache__ python/src/ml_worker/__pycache__

# Start the ML worker
run-ml:
	cd python && uv run python -m ml_worker.server

# Run the CLI in interactive mode
run-cli:
	cargo run --release -p graphrag-cli -- interactive

# Initial setup
setup:
	./setup.sh

# Add a test note
demo-add:
	cargo run --release -p graphrag-cli -- --memory add "Machine learning is transforming how we build software" --title "ML Overview"
	cargo run --release -p graphrag-cli -- --memory add "Neural networks learn hierarchical representations of data" --title "Neural Networks"
	cargo run --release -p graphrag-cli -- --memory add "Transformers use attention mechanisms for sequence modeling" --title "Transformers"

# Demo search
demo-search:
	cargo run --release -p graphrag-cli -- --memory search "how do neural networks work"

# Show stats
stats:
	cargo run --release -p graphrag-cli -- stats

# Format code
fmt:
	cargo fmt
	cd python && uv run ruff format src tests

# Lint
lint:
	cargo clippy
	cd python && uv run ruff check src tests

# Help
help:
	@echo "GraphRAG Notes - Makefile targets"
	@echo ""
	@echo "  build              Build release binary"
	@echo "  test               Run all tests"
	@echo "  test-rust          Run Rust unit tests"
	@echo "  test-rust-integration  Run Rust integration tests (needs ML worker)"
	@echo "  test-python        Run Python tests"
	@echo "  run-ml             Start the ML worker"
	@echo "  run-cli            Start interactive CLI"
	@echo "  setup              Initial setup (Python venv + Rust build)"
	@echo "  clean              Clean build artifacts"
	@echo "  fmt                Format code"
	@echo "  lint               Run linters"
