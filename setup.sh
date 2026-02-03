#!/bin/bash
# Setup script for GraphRAG Notes

set -e

echo "🚀 Setting up GraphRAG Notes..."

# Check prerequisites
command -v cargo >/dev/null 2>&1 || {
    echo "❌ Rust not found. Install from https://rustup.rs/"
    exit 1
}

command -v uv >/dev/null 2>&1 || {
    echo "❌ uv not found. Install from https://github.com/astral-sh/uv"
    exit 1
}

# Setup Python environment
echo "📦 Setting up Python ML worker..."
cd python
uv sync
cd ..

echo "🔨 Building Rust CLI..."
cargo build --release

echo ""
echo "✅ Setup complete!"
echo ""
echo "To get started:"
echo ""
echo "1. Start the ML worker (in one terminal):"
echo "   cd python && uv run python -m ml_worker.server"
echo ""
echo "2. Use the CLI (in another terminal):"
echo "   ./target/release/graphrag --help"
echo ""
echo "   Or add some notes:"
echo "   ./target/release/graphrag add \"Your first note\""
echo ""
echo "   Search:"
echo "   ./target/release/graphrag search \"your query\""
echo ""
