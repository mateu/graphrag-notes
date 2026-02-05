#!/bin/bash
# Setup script for GraphRAG Notes

set -e

echo "🚀 Setting up GraphRAG Notes..."

# Install system deps needed for Rust builds (openssl + clang for bindgen)
OS_NAME="$(uname -s)"
if [ "$OS_NAME" = "Linux" ]; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "📦 Installing Linux build dependencies..."
        sudo apt-get update
        sudo apt-get install -y pkg-config libssl-dev clang libclang-dev
    else
        echo "⚠️ Unsupported Linux package manager. Please install: pkg-config, libssl-dev (or openssl-devel), clang, libclang-dev."
    fi
elif [ "$OS_NAME" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
        echo "📦 Installing macOS build dependencies..."
        brew install pkg-config openssl@3 llvm
        echo "ℹ️ If builds fail, you may need: export LIBCLANG_PATH=\"$(brew --prefix llvm)/lib\""
    else
        echo "⚠️ Homebrew not found. Please install: pkg-config, openssl@3, llvm."
    fi
fi

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
FORCE_BUILD=0
if [ "${1:-}" = "--force" ]; then
    FORCE_BUILD=1
fi

if [ -f "target/release/graphrag" ] && [ "$FORCE_BUILD" -eq 0 ]; then
    echo "✅ Release binary already exists. Skipping build (use --force to rebuild)."
else
    cargo build --release
fi

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
