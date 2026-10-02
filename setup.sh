#!/usr/bin/env bash
# Explicit source-build fallback. This never installs packages, models, or configuration.
set -euo pipefail

case "${1:-}" in
    -h|--help)
        printf 'Usage: ./setup.sh [--force]\nBuild the locked graphrag CLI from source. --force is accepted for compatibility; Cargo always checks for changes.\n'
        exit 0 ;;
    ''|--force) ;;
    *) printf 'Unknown option: %s (use --help)\n' "$1" >&2; exit 1 ;;
esac
[ "$#" -le 1 ] || { printf 'Expected at most one option (use --help)\n' >&2; exit 1; }

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if ! command -v cargo >/dev/null 2>&1; then
    printf 'Rust is required. Install rustup from https://rustup.rs, then run ./setup.sh again.\n' >&2
    exit 1
fi

# The repository enables sccache, but it is optional for a first source build.
if [ -z "${RUSTC_WRAPPER+x}" ] && ! command -v sccache >/dev/null 2>&1; then
    export RUSTC_WRAPPER=''
fi

if [ "$(uname -s)" = Darwin ] && command -v brew >/dev/null 2>&1; then
    llvm_prefix="$(brew --prefix llvm 2>/dev/null || true)"
    if [ -n "$llvm_prefix" ] && [ -d "$llvm_prefix/lib" ]; then
        export LIBCLANG_PATH="${LIBCLANG_PATH:-$llvm_prefix/lib}"
    fi
    openssl_prefix="$(brew --prefix openssl@3 2>/dev/null || true)"
    if [ -n "$openssl_prefix" ] && [ -d "$openssl_prefix/lib/pkgconfig" ]; then
        export PKG_CONFIG_PATH="$openssl_prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    fi
fi

missing=''
for dependency in clang cmake pkg-config; do
    if ! command -v "$dependency" >/dev/null 2>&1; then missing="$missing $dependency"; fi
done
if [ -n "$missing" ]; then
    printf 'Missing source-build prerequisites:%s\n' "$missing" >&2
    if [ "$(uname -s)" = Darwin ]; then
        printf 'Install the Xcode command-line tools (xcode-select --install) and Homebrew packages: brew install cmake pkg-config openssl@3 llvm\n' >&2
    else
        printf 'On Debian/Ubuntu: sudo apt-get install build-essential cmake pkg-config libssl-dev clang libclang-dev\n' >&2
    fi
    exit 1
fi
if [ "$(uname -s)" = Linux ] && ! pkg-config --exists openssl; then
    printf 'OpenSSL development headers are missing. On Debian/Ubuntu: sudo apt-get install libssl-dev\n' >&2
    exit 1
fi

printf 'Building graphrag from the locked dependency versions (the first build can take a while)...\n'
cargo build --locked --release -p graphrag-cli --bin graphrag
printf '\nBuilt %s/target/release/graphrag\n' "$project_dir"
printf 'Add %s/target/release to your PATH, then run: graphrag init --backend ollama\n' "$project_dir"
printf 'Starter sample: %s/samples/first-notes.md\n' "$project_dir"
