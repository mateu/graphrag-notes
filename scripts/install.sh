#!/usr/bin/env bash
# Install a published release without Cargo, package managers, or inference services.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash scripts/install.sh [--version VERSION] [--bin-dir DIR] [--data-dir DIR] [--force]

Download the matching macOS/Linux release and verify its SHA-256 checksum.
VERSION accepts 0.1.0 or v0.1.0; omitted means the latest published release.
Defaults: $HOME/.local/bin/graphrag and $HOME/.local/share/graphrag-notes/samples/first-notes.md
--force replaces an existing binary. Existing sample notes are always preserved.
USAGE
}

fail() { printf 'Install failed: %s\n' "$*" >&2; exit 1; }

version=''
bin_dir="${HOME:?HOME must be set}/.local/bin"
data_dir="${HOME}/.local/share/graphrag-notes"
force=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --version|--bin-dir|--data-dir)
            [ "$#" -ge 2 ] && [ -n "$2" ] || fail "$1 requires a value"
            case "$1" in
                --version) version="$2" ;;
                --bin-dir) bin_dir="$2" ;;
                --data-dir) data_dir="$2" ;;
            esac
            shift 2 ;;
        --force) force=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown option $1 (use --help)" ;;
    esac
done

command -v curl >/dev/null 2>&1 || fail 'curl is required to download releases'
command -v tar >/dev/null 2>&1 || fail 'tar is required to unpack releases'
if command -v sha256sum >/dev/null 2>&1; then
    checksum() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    checksum() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    fail 'sha256sum or shasum is required to verify downloads'
fi

case "$(uname -s)/$(uname -m)" in
    Darwin/arm64|Darwin/aarch64) target='aarch64-apple-darwin' ;;
    Darwin/x86_64) target='x86_64-apple-darwin' ;;
    Linux/x86_64) target='x86_64-unknown-linux-gnu' ;;
    *) fail 'no binary release for this platform; follow the source-build instructions in README.md' ;;
esac

releases='https://github.com/mateu/graphrag-notes/releases'
if [ -z "$version" ]; then
    latest_url="$(curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL \
        -o /dev/null -w '%{url_effective}' "$releases/latest")" || \
        fail 'no published release could be found; use ./setup.sh from a source checkout, or specify --version'
    case "$latest_url" in
        "$releases/tag/"*) version="${latest_url##*/}" ;;
        *) fail 'no published release could be found; use ./setup.sh from a source checkout' ;;
    esac
fi
version="${version#v}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]] || \
    fail 'version must be a release version such as 0.1.0 or v0.1.0'
tag="v$version"
asset="graphrag-notes-$tag-$target.tar.gz"

[ ! -L "$bin_dir/graphrag" ] || fail "$bin_dir/graphrag is a symlink; choose another --bin-dir"
if [ -e "$bin_dir/graphrag" ] && [ "$force" -ne 1 ]; then
    fail "$bin_dir/graphrag already exists; use --force to replace it"
fi
[ ! -d "$bin_dir/graphrag" ] || fail "$bin_dir/graphrag is a directory"

temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/graphrag-install.XXXXXX")"
binary_temp=''
sample_temp=''
cleanup() {
    rm -rf "$temp_dir"
    if [ -n "$binary_temp" ]; then rm -f "$binary_temp"; fi
    if [ -n "$sample_temp" ]; then rm -f "$sample_temp"; fi
}
trap cleanup EXIT

printf 'Downloading %s\n' "$asset"
for file in SHA256SUMS "$asset"; do
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL \
        "$releases/download/$tag/$file" -o "$temp_dir/$file" || \
        fail "cannot download $file for $tag; check the release, or use ./setup.sh from source"
done
expected="$(awk -v asset="$asset" '$2 == asset || $2 == "*" asset {print $1}' "$temp_dir/SHA256SUMS")"
[[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]] || fail 'release manifest must contain exactly one checksum for this archive'
actual="$(checksum "$temp_dir/$asset")"
[ "$(printf '%s' "$actual" | tr '[:upper:]' '[:lower:]')" = \
    "$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]')" ] || fail 'archive checksum does not match; nothing was installed'

# Reject traversal, extra payloads, and links before extraction, even after checksum verification.
tar -tzf "$temp_dir/$asset" > "$temp_dir/contents"
awk '$0 != "graphrag" && $0 != "samples/" && $0 != "samples/first-notes.md" {exit 1}' \
    "$temp_dir/contents" || fail 'archive contains an unexpected path'
tar -tvzf "$temp_dir/$asset" | awk 'substr($1, 1, 1) != "-" && substr($1, 1, 1) != "d" {exit 1}' || \
    fail 'archive contains a link or unsupported file type'
mkdir "$temp_dir/unpacked"
tar -xzf "$temp_dir/$asset" -C "$temp_dir/unpacked"
[ -f "$temp_dir/unpacked/graphrag" ] && [ -f "$temp_dir/unpacked/samples/first-notes.md" ] || \
    fail 'archive is missing the binary or starter sample'

mkdir -p "$bin_dir" "$data_dir/samples"
# Stage on the same filesystem so a failed download or copy cannot truncate the old binary.
binary_temp="$(mktemp "$bin_dir/.graphrag.XXXXXX")"
cp "$temp_dir/unpacked/graphrag" "$binary_temp"
chmod 755 "$binary_temp"
if [ ! -e "$data_dir/samples/first-notes.md" ] && [ ! -L "$data_dir/samples/first-notes.md" ]; then
    sample_temp="$(mktemp "$data_dir/samples/.first-notes.XXXXXX")"
    cp "$temp_dir/unpacked/samples/first-notes.md" "$sample_temp"
    chmod 644 "$sample_temp"
    # A simultaneous installation or editor creating the sample wins without being overwritten.
    if ! ln -n "$sample_temp" "$data_dir/samples/first-notes.md" 2>/dev/null; then
        [ -e "$data_dir/samples/first-notes.md" ] || [ -L "$data_dir/samples/first-notes.md" ] || \
            fail 'could not publish the starter sample'
    fi
    rm -f "$sample_temp"
    sample_temp=''
fi
if [ "$force" -eq 1 ]; then
    mv -f "$binary_temp" "$bin_dir/graphrag"
else
    # Link publication is atomic and never replaces a binary created after the initial check.
    ln -n "$binary_temp" "$bin_dir/graphrag" 2>/dev/null || \
        fail "$bin_dir/graphrag appeared during installation; use --force to replace it"
    rm -f "$binary_temp"
fi
binary_temp=''

printf 'Installed %s to %s/graphrag\n' "$tag" "$bin_dir"
printf 'Starter sample: %s/samples/first-notes.md\n' "$data_dir"
case ":${PATH:-}:" in
    *":$bin_dir:"*) ;;
    *) printf 'Add this directory to your PATH: %s\n' "$bin_dir" ;;
esac
printf 'Next: %s/graphrag init --backend ollama\n' "$bin_dir"
