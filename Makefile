.PHONY: all build test clean run-cli setup demo-add demo-search stats fmt lint help

all: build

build:
	cargo build --release

build-debug:
	cargo build

test:
	cargo test --offline --locked

clean:
	cargo clean

run-cli:
	cargo run --release -p graphrag-cli -- interactive

setup:
	./setup.sh

demo-add:
	cargo run --release -p graphrag-cli -- --memory add "Machine learning is transforming how we build software" --title "ML Overview"
	cargo run --release -p graphrag-cli -- --memory add "Neural networks learn hierarchical representations of data" --title "Neural Networks"
	cargo run --release -p graphrag-cli -- --memory add "Transformers use attention mechanisms for sequence modeling" --title "Transformers"

demo-search:
	cargo run --release -p graphrag-cli -- --memory search "how do neural networks work"

stats:
	cargo run --release -p graphrag-cli -- stats

fmt:
	cargo fmt

lint:
	cargo clippy

help:
	@echo "GraphRAG Notes - Makefile targets"
	@echo ""
	@echo "  build        Build release binary"
	@echo "  build-debug  Build debug binary"
	@echo "  test         Run Rust tests"
	@echo "  run-cli      Start interactive CLI"
	@echo "  setup        Build-local setup helper"
	@echo "  clean        Clean build artifacts"
	@echo "  fmt          Format Rust code"
	@echo "  lint         Run clippy"
