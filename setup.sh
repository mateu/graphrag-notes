#!/bin/bash
set -e

echo "🚀 Setting up GraphRAG Notes..."

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

if ! command -v cargo >/dev/null 2>&1; then
    echo "❌ Rust not found. Installing via rustup from https://rustup.rs/ ..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    if [ -f "$HOME/.cargo/env" ]; then
        . "$HOME/.cargo/env"
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        echo "❌ Rust installation via rustup appears to have failed. Please install manually from https://rustup.rs/"
        exit 1
    fi
fi

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
echo "Next steps:"
echo "1. Start inference backends (TEI/TGI via docker compose, or Ollama)"
echo "2. Run: ./target/release/graphrag --help"
echo ""
echo "Default backend endpoints:"
echo "  TEI_URL=http://localhost:8081"
echo "  TGI_URL=http://localhost:8082"
echo ""
echo "Ollama alternative:"
echo "  export TEI_PROVIDER=ollama"
echo "  export TGI_PROVIDER=ollama"
