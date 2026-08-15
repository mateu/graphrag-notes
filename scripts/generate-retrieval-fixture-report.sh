#!/usr/bin/env bash
# Generate the deterministic, offline retrieval-fixture candidate accepted by
# update-retrieval-baseline.sh. Refuse to overwrite an existing candidate so
# baseline review always begins with an intentional new artifact.
set -euo pipefail

usage() {
  echo "usage: $0 OUTPUT_REPORT.json" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
output=$1
[[ ! -e "$output" ]] || {
  echo "refusing to overwrite existing fixture report: $output" >&2
  exit 2
}
parent=$(dirname "$output")
[[ -d "$parent" ]] || {
  echo "output directory does not exist: $parent" >&2
  exit 2
}

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
GRAPHRAG_RETRIEVAL_FIXTURE_REPORT="$output" \
  cargo test -p graphrag-cli eval::tests::writes_retrieval_fixture_report_when_requested \
    --bin graphrag --locked -- --exact

[[ -s "$output" ]] || {
  echo "fixture report producer did not write a non-empty report: $output" >&2
  exit 1
}
echo "Wrote deterministic retrieval fixture report: $output"
