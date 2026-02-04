#!/bin/bash
# Setup script for GraphRAG Notes

set -e

echo "🚀 Setting up GraphRAG Notes..."

# Check/install prerequisites (portable across macOS and Linux)
if ! command -v cargo >/dev/null 2>&1; then
    echo "❌ Rust not found. Installing via rustup from https://rustup.rs/ ..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1090
    if [ -f "$HOME/.cargo/env" ]; then
        . "$HOME/.cargo/env"
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        echo "❌ Rust installation via rustup appears to have failed. Please install manually from https://rustup.rs/"
        exit 1
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "❌ uv not found. Installing via https://astral.sh/uv/install.sh ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installer typically adjusts PATH via shell rc; try a rehash
    if ! command -v uv >/dev/null 2>&1; then
        echo "⚠️ uv was installed but is not yet on PATH. Please restart your shell or source your shell rc file, then re-run ./setup.sh or make setup."
        exit 1
    fi
fi

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
