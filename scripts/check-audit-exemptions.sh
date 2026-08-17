#!/usr/bin/env bash
set -euo pipefail

target="${1:-x86_64-unknown-linux-gnu}"
dependency_tree="$(cargo tree --locked --target "$target" -e normal,build --prefix none)"

for crate in rkyv rsa; do
    if grep -Eq "^${crate} v" <<<"$dependency_tree"; then
        echo "audit exemption is no longer safe: ${crate} is active for ${target}" >&2
        exit 1
    fi
done

echo "audit exemptions remain unreachable for ${target}"
