#!/usr/bin/env bash
set -euo pipefail

target="${1:-x86_64-unknown-linux-gnu}"
dependency_tree="$(cargo tree --locked --target "$target" -e normal,build --prefix none)"

# RUSTSEC-2026-0235 applies to rkyv 0.7 only; SurrealDB 3.3 uses the
# non-affected 0.8 series for its native full-text dictionary dependency.
for crate in "rkyv v0.7" rsa crypto-bigint; do
    if grep -Eq "^${crate}" <<<"$dependency_tree"; then
        echo "audit exemption is no longer safe: ${crate} is active for ${target}" >&2
        exit 1
    fi
done

# RUSTSEC-2026-0454 is reachable but temporary (#246): once the SurrealDB 4.x
# upgrade drops reblessive, the exemption must be removed rather than linger.
if ! grep -Eq "^reblessive v0\.4" <<<"$dependency_tree"; then
    echo "RUSTSEC-2026-0454 exemption is stale: reblessive 0.4 is gone; remove it (#246)" >&2
    exit 1
fi

echo "audit exemptions remain unreachable for ${target}"
