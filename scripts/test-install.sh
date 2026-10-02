#!/usr/bin/env bash
# Deterministic installer and source-build checks; no network or Rust compilation.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/graphrag-install-test.XXXXXX")"
trap 'rm -rf "$test_dir"' EXIT
mkdir -p "$test_dir/tools" "$test_dir/release" "$test_dir/staging/samples" "$test_dir/home"
export TEST_RELEASE_DIR="$test_dir/release"
export TEST_REQUEST_LOG="$test_dir/requests"
export TEST_OS=Linux TEST_ARCH=x86_64
export REAL_UNAME
REAL_UNAME="$(command -v uname)"
export REAL_LN
REAL_LN="$(command -v ln)"

cat > "$test_dir/tools/uname" <<'TOOL'
#!/usr/bin/env bash
case "$1" in
    -s) printf '%s\n' "$TEST_OS" ;;
    -m) printf '%s\n' "$TEST_ARCH" ;;
    *) exec "$REAL_UNAME" "$@" ;;
esac
TOOL
cat > "$test_dir/tools/curl" <<'TOOL'
#!/usr/bin/env bash
set -euo pipefail
output='' url='' secure=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --proto|--proto-redir)
            [ "$2" = '=https' ] || exit 2
            secure=$((secure + 1)); shift 2 ;;
        --tlsv1.2) secure=$((secure + 1)); shift ;;
        -fsSL) shift ;;
        -o) output="$2"; shift 2 ;;
        -w) [ "$2" = '%{url_effective}' ] || exit 2; shift 2 ;;
        https://*) url="$1"; shift ;;
        *) exit 2 ;;
    esac
done
[ "$secure" = 3 ] || exit 2
printf '%s\n' "$url" >> "$TEST_REQUEST_LOG"
case "$url" in
    https://github.com/mateu/graphrag-notes/releases/latest)
        [ "${TEST_NO_RELEASE:-0}" != 1 ] || exit 22
        printf 'https://github.com/mateu/graphrag-notes/releases/tag/v0.1.0' ;;
    https://github.com/mateu/graphrag-notes/releases/download/v0.1.0/*)
        [ -f "$TEST_RELEASE_DIR/${url##*/}" ] || exit 22
        cp "$TEST_RELEASE_DIR/${url##*/}" "$output" ;;
    *) exit 22 ;;
esac
TOOL
chmod +x "$test_dir/tools/uname" "$test_dir/tools/curl"
cat > "$test_dir/tools/ln" <<'TOOL'
#!/usr/bin/env bash
# Introduce a deterministic concurrent write immediately before atomic publication.
if [ "${TEST_RACE_BINARY:-}" = "$3" ]; then printf 'concurrent binary\n' > "$3"; fi
if [ "${TEST_RACE_SAMPLE:-}" = "$3" ]; then printf 'concurrent sample\n' > "$3"; fi
exec "$REAL_LN" "$@"
TOOL
chmod +x "$test_dir/tools/ln"

printf '#!/bin/sh\nprintf "fixture graphrag\\n"\n' > "$test_dir/staging/graphrag"
chmod +x "$test_dir/staging/graphrag"
printf '# First notes\nFixture sample.\n' > "$test_dir/staging/samples/first-notes.md"
for target in aarch64-apple-darwin x86_64-apple-darwin x86_64-unknown-linux-gnu; do
    tar -czf "$TEST_RELEASE_DIR/graphrag-notes-v0.1.0-$target.tar.gz" \
        -C "$test_dir/staging" graphrag samples/first-notes.md
done
manifest() {
    : > "$TEST_RELEASE_DIR/SHA256SUMS"
    for file in "$TEST_RELEASE_DIR"/*.tar.gz; do
        if command -v sha256sum >/dev/null 2>&1; then
            hash="$(sha256sum "$file" | awk '{print $1}')"
        else
            hash="$(shasum -a 256 "$file" | awk '{print $1}')"
        fi
        printf '%s  %s\n' "$hash" "${file##*/}" >> "$TEST_RELEASE_DIR/SHA256SUMS"
    done
}
manifest
cp "$TEST_RELEASE_DIR/SHA256SUMS" "$test_dir/valid-checksums"
installer() {
    HOME="$test_dir/home" PATH="$test_dir/tools:$PATH" bash "$project_dir/scripts/install.sh" "$@" > "$test_dir/output" 2>&1
}
expect_failure() {
    message="$1"; shift
    if installer "$@"; then
        printf 'Expected install failure: %s\n' "$message" >&2
        exit 1
    fi
    grep -F "$message" "$test_dir/output" >/dev/null || { cat "$test_dir/output"; exit 1; }
}

# Defaults resolve latest, verify download, and provide the bundled sample.
installer
cmp "$test_dir/staging/graphrag" "$test_dir/home/.local/bin/graphrag"
cmp "$test_dir/staging/samples/first-notes.md" "$test_dir/home/.local/share/graphrag-notes/samples/first-notes.md"
[ -x "$test_dir/home/.local/bin/graphrag" ]
grep -F '/releases/latest' "$TEST_REQUEST_LOG" >/dev/null
printf 'PASS: latest release, checksum, binary permissions, and starter sample\n'

# Explicit versions work on both macOS architectures, with paths containing spaces.
for arch in arm64 x86_64; do
    TEST_OS=Darwin TEST_ARCH="$arch" installer --version v0.1.0 \
        --bin-dir "$test_dir/mac $arch/bin" --data-dir "$test_dir/mac $arch/data"
    cmp "$test_dir/staging/graphrag" "$test_dir/mac $arch/bin/graphrag"
done
printf 'PASS: macOS ARM and Intel selection, explicit tag, and custom paths\n'

expect_failure 'already exists' --version 0.1.0
printf 'my edited sample\n' > "$test_dir/home/.local/share/graphrag-notes/samples/first-notes.md"
printf 'existing configuration\n' > "$test_dir/home/.local/share/graphrag-notes/config.toml"
installer --version 0.1.0 --force
grep -Fx 'my edited sample' "$test_dir/home/.local/share/graphrag-notes/samples/first-notes.md" >/dev/null
grep -Fx 'existing configuration' "$test_dir/home/.local/share/graphrag-notes/config.toml" >/dev/null
printf 'PASS: overwrite requires --force and preserves edited sample/configuration\n'

TEST_RACE_BINARY="$test_dir/concurrent-bin/graphrag" expect_failure 'appeared during installation' \
    --version 0.1.0 --bin-dir "$test_dir/concurrent-bin" --data-dir "$test_dir/concurrent-data"
grep -Fx 'concurrent binary' "$test_dir/concurrent-bin/graphrag" >/dev/null
TEST_RACE_SAMPLE="$test_dir/concurrent-sample/samples/first-notes.md" installer \
    --version 0.1.0 --bin-dir "$test_dir/concurrent-sample-bin" --data-dir "$test_dir/concurrent-sample"
grep -Fx 'concurrent sample' "$test_dir/concurrent-sample/samples/first-notes.md" >/dev/null
cmp "$test_dir/staging/graphrag" "$test_dir/concurrent-sample-bin/graphrag"
printf 'PASS: concurrent binary and sample writes are preserved atomically\n'

printf '%064d  graphrag-notes-v0.1.0-x86_64-unknown-linux-gnu.tar.gz\n' 0 > "$TEST_RELEASE_DIR/SHA256SUMS"
expect_failure 'checksum does not match' --version 0.1.0 --force
cmp "$test_dir/staging/graphrag" "$test_dir/home/.local/bin/graphrag"
cat "$test_dir/valid-checksums" "$test_dir/valid-checksums" > "$TEST_RELEASE_DIR/SHA256SUMS"
expect_failure 'exactly one checksum' --version 0.1.0 --force
cp "$test_dir/valid-checksums" "$TEST_RELEASE_DIR/SHA256SUMS"
printf 'PASS: checksum mismatch and duplicate entries cannot replace installed binary\n'

TEST_NO_RELEASE=1 expect_failure 'no published release' --bin-dir "$test_dir/no-release"
expect_failure 'cannot download' --version 9.9.9 --bin-dir "$test_dir/no-release"
TEST_OS=Linux TEST_ARCH=aarch64 expect_failure 'no binary release' --version 0.1.0 --bin-dir "$test_dir/no-platform"
expect_failure 'version must be' --version '../0.1.0' --bin-dir "$test_dir/bad-version"
expect_failure 'requires a value' --version
[ ! -e "$test_dir/no-release/graphrag" ]
printf 'PASS: missing release, unsupported platform, and malformed input give recovery guidance\n'

mkdir "$test_dir/symlink-bin"
ln -s "$test_dir/staging/graphrag" "$test_dir/symlink-bin/graphrag"
expect_failure 'is a symlink' --version 0.1.0 --force --bin-dir "$test_dir/symlink-bin"
cmp "$test_dir/staging/graphrag" "$test_dir/home/.local/bin/graphrag"
printf 'PASS: an existing binary symlink is preserved\n'

printf 'unexpected file\n' > "$test_dir/staging/unexpected"
tar -czf "$TEST_RELEASE_DIR/graphrag-notes-v0.1.0-x86_64-unknown-linux-gnu.tar.gz" \
    -C "$test_dir/staging" graphrag samples/first-notes.md unexpected
manifest
expect_failure 'unexpected path' --version 0.1.0 --force
mv "$test_dir/staging/graphrag" "$test_dir/staging/graphrag-real"
ln -s graphrag-real "$test_dir/staging/graphrag"
tar -czf "$TEST_RELEASE_DIR/graphrag-notes-v0.1.0-x86_64-unknown-linux-gnu.tar.gz" \
    -C "$test_dir/staging" graphrag samples/first-notes.md
manifest
expect_failure 'link or unsupported file type' --version 0.1.0 --force
cmp "$test_dir/staging/graphrag-real" "$test_dir/home/.local/bin/graphrag"
printf 'PASS: unexpected archive entries and symbolic links are rejected before installation\n'

# Isolate setup from real Cargo, Homebrew, sccache, and package managers.
mkdir "$test_dir/build-tools"
for tool in bash dirname pwd; do
    ln -s "$(command -v "$tool")" "$test_dir/build-tools/$tool"
done
cp "$test_dir/tools/uname" "$test_dir/build-tools/uname"
for tool in clang cmake pkg-config; do
    printf '#!/bin/sh\nexit 0\n' > "$test_dir/build-tools/$tool"
    chmod +x "$test_dir/build-tools/$tool"
done
cat > "$test_dir/build-tools/cargo" <<'TOOL'
#!/bin/sh
printf '%s\n' "$*" > "$TEST_BUILD_LOG"
printf '%s' "${RUSTC_WRAPPER-unset}" > "$TEST_WRAPPER_LOG"
TOOL
chmod +x "$test_dir/build-tools/cargo"
export TEST_BUILD_LOG="$test_dir/build.log" TEST_WRAPPER_LOG="$test_dir/wrapper.log"
(unset RUSTC_WRAPPER; TEST_OS=Linux PATH="$test_dir/build-tools" bash "$project_dir/setup.sh") > "$test_dir/setup-output"
grep -Fx 'build --locked --release -p graphrag-cli --bin graphrag' "$TEST_BUILD_LOG" >/dev/null
[ ! -s "$TEST_WRAPPER_LOG" ]
RUSTC_WRAPPER=/custom/compiler-cache TEST_OS=Linux PATH="$test_dir/build-tools" bash "$project_dir/setup.sh" > "$test_dir/setup-output"
grep -Fx '/custom/compiler-cache' "$TEST_WRAPPER_LOG" >/dev/null
rm "$test_dir/build-tools/clang"
if TEST_OS=Linux PATH="$test_dir/build-tools" bash "$project_dir/setup.sh" > "$test_dir/setup-output" 2>&1; then
    printf 'Expected setup to report missing clang\n' >&2
    exit 1
fi
grep -F 'Missing source-build prerequisites: clang' "$test_dir/setup-output" >/dev/null
grep -F 'sudo apt-get install' "$test_dir/setup-output" >/dev/null
printf 'PASS: source build is locked, supports optional sccache, and reports missing prerequisites\n'
