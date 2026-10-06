#!/usr/bin/env bash
# Exercise the installer with real local assets and doubled HTTPS transport.
# No network, Cargo, configuration/database writes, GUI, or inference calls.
set -euo pipefail
if [ "$#" -ne 3 ]; then
    printf 'Usage: bash scripts/verify-local-release.sh DIST VERSION EVIDENCE_FILE\n' >&2
    exit 2
fi
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release_dist="$(cd "$1" && pwd)"
release_version="${2#v}"
evidence_file="$3"
[[ "$release_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]] || exit 2
[ ! -e "$evidence_file" ] && [ ! -L "$evidence_file" ] || { printf 'Evidence already exists\n' >&2; exit 1; }
case "$(uname -s)/$(uname -m)" in
    Darwin/arm64|Darwin/aarch64) release_target='aarch64-apple-darwin' ;;
    Darwin/x86_64) release_target='x86_64-apple-darwin' ;;
    Linux/x86_64) release_target='x86_64-unknown-linux-gnu' ;;
    *) printf 'Unsupported native test platform\n' >&2; exit 1 ;;
esac
asset="graphrag-notes-v$release_version-$release_target.tar.gz"
for required in "$asset" SHA256SUMS BUILDINFO.json; do
    [ -f "$release_dist/$required" ] && [ ! -L "$release_dist/$required" ] || { printf 'Missing regular release asset: %s\n' "$required" >&2; exit 1; }
done
temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/graphrag-local-asset.XXXXXX")"
trap 'rm -rf "$temp_dir"' EXIT
mkdir "$temp_dir/tools" "$temp_dir/home" "$temp_dir/unpacked"
export GRN_RELEASE_DIST="$release_dist" GRN_RELEASE_TAG="v$release_version" GRN_RELEASE_REQUESTS="$temp_dir/requests"
cat > "$temp_dir/tools/curl" <<'TOOL'
#!/bin/bash
set -euo pipefail
output='' url='' secure=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --proto|--proto-redir) [ "$2" = '=https' ] || exit 2; secure=$((secure + 1)); shift 2 ;;
        --tlsv1.2) secure=$((secure + 1)); shift ;;
        -fsSL) shift ;;
        -o) output="$2"; shift 2 ;;
        https://*) url="$1"; shift ;;
        *) exit 2 ;;
    esac
done
[ "$secure" = 3 ] && [ -n "$output" ] || exit 2
case "$url" in
    "https://github.com/mateu/graphrag-notes/releases/download/$GRN_RELEASE_TAG/"*)
        name="${url##*/}"
        case "$name" in SHA256SUMS|BUILDINFO.json|BUILDINFO-*.json|BUILDINFO.identity|BUILDINFO-*.identity|graphrag-notes-*.tar.gz) ;; *) exit 22 ;; esac
        [ -f "$GRN_RELEASE_DIST/$name" ] && [ ! -L "$GRN_RELEASE_DIST/$name" ] || exit 22
        printf '%s\n' "$url" >> "$GRN_RELEASE_REQUESTS"
        cp "$GRN_RELEASE_DIST/$name" "$output" ;;
    *) exit 22 ;;
esac
TOOL
chmod +x "$temp_dir/tools/curl"
# Installation must remain usable without a Python runtime. Measurement below
# uses an absolute interpreter separately from the installer's PATH.
cat > "$temp_dir/tools/python3" <<'TOOL'
#!/bin/sh
printf 'Installer unexpectedly invoked Python\n' >&2
exit 99
TOOL
chmod +x "$temp_dir/tools/python3"
installer() {
    HOME="$temp_dir/home" PATH="$temp_dir/tools:/usr/bin:/bin:/usr/sbin:/sbin" \
        bash "$project_dir/scripts/install.sh" --version "$release_version" "$@"
}
measurement_python="$(command -v python3)"
install_started="$("$measurement_python" -c 'import time; print(time.monotonic_ns())')"
if ! installer > "$temp_dir/install.log" 2>&1; then
    cat "$temp_dir/install.log" >&2
    exit 1
fi
install_elapsed="$("$measurement_python" -c 'import sys,time; print(round((time.monotonic_ns()-int(sys.argv[1]))/1e9,6))' "$install_started")"
tar -xzf "$release_dist/$asset" -C "$temp_dir/unpacked"
cmp "$temp_dir/unpacked/graphrag" "$temp_dir/home/.local/bin/graphrag"
cmp "$temp_dir/unpacked/samples/first-notes.md" "$temp_dir/home/.local/share/graphrag-notes/samples/first-notes.md"
if [ -d "$temp_dir/unpacked/release" ]; then
    diff -r "$temp_dir/unpacked/release" "$temp_dir/home/.local/share/graphrag-notes/releases/v$release_version" >/dev/null
fi
[ -x "$temp_dir/home/.local/bin/graphrag" ]
HOME="$temp_dir/home" "$temp_dir/home/.local/bin/graphrag" --version | grep -Fx "graphrag $release_version" >/dev/null
if installer > "$temp_dir/refusal.log" 2>&1; then printf 'Expected existing-binary refusal\n' >&2; exit 1; fi
grep -F 'already exists' "$temp_dir/refusal.log" >/dev/null
printf 'operator edited sample\n' > "$temp_dir/home/.local/share/graphrag-notes/samples/first-notes.md"
printf 'operator edited configuration\n' > "$temp_dir/home/.local/share/graphrag-notes/config.toml"
if ! installer --force > "$temp_dir/reinstall.log" 2>&1; then
    cat "$temp_dir/reinstall.log" >&2
    exit 1
fi
grep -Fx 'operator edited sample' "$temp_dir/home/.local/share/graphrag-notes/samples/first-notes.md" >/dev/null
grep -Fx 'operator edited configuration' "$temp_dir/home/.local/share/graphrag-notes/config.toml" >/dev/null
cmp "$temp_dir/unpacked/graphrag" "$temp_dir/home/.local/bin/graphrag"
if command -v shasum >/dev/null 2>&1; then
    digest() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    digest() { sha256sum "$1" | awk '{print $1}'; }
fi
archive_hash="$(digest "$release_dist/$asset")"
binary_hash="$(digest "$temp_dir/home/.local/bin/graphrag")"
sample_hash="$(digest "$temp_dir/unpacked/samples/first-notes.md")"
metadata_hash="$(awk '$2 == "BUILDINFO.json" {print $1}' "$release_dist/SHA256SUMS")"
[[ "$metadata_hash" =~ ^[0-9a-f]{64}$ ]] && [ "$metadata_hash" = "$(digest "$release_dist/BUILDINFO.json")" ] || \
    { printf 'BUILDINFO checksum differs from manifest\n' >&2; exit 1; }
metadata_value() {
    "$measurement_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
        "$release_dist/BUILDINFO.json" "$1"
}
source_commit="$(metadata_value source_commit)"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] && \
    [ "$(metadata_value version)" = "$release_version" ] && \
    [ "$(metadata_value binary_sha256)" = "$binary_hash" ] && \
    [ "$(metadata_value sample_sha256)" = "$sample_hash" ] && \
    [ "$(metadata_value archive_sha256)" = "$archive_hash" ] || \
    { printf 'Installed payload differs from BUILDINFO provenance\n' >&2; exit 1; }
# An exclusive write preserves earlier evidence; JSON values below are validated
# version/target/hash tokens, never arbitrary command output or host paths.
(set -o noclobber; cat > "$evidence_file" <<JSON
{
  "status": "passed",
  "transport": "local curl fixture; published download not tested",
  "version": "$release_version",
  "source_commit": "$source_commit",
  "target": "$release_target",
  "archive_sha256": "$archive_hash",
  "binary_sha256": "$binary_hash",
  "sample_sha256": "$sample_hash",
  "elapsed_seconds": $install_elapsed,
  "fresh_install_invocations": 1,
  "installer_invocations": 3,
  "checks": ["fresh native-tools install without Python", "exact version", "payload hash equality", "existing-binary refusal", "explicit force reinstall", "edited-sample/config preservation", "matching versioned client/document bundle"]
}
JSON
)
printf 'PASS: real local assets installed, refused overwrite, and preserved edited sample/config\n'
printf 'Transport was doubled locally; published HTTPS installation remains untested.\n'
