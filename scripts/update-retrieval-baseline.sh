#!/usr/bin/env bash
# Review and (only after an explicit interactive acknowledgement) replace the
# versioned retrieval baseline. CI never invokes this script.
set -euo pipefail

usage() {
  echo "usage: $0 CANDIDATE_REPORT.json [--apply]" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
candidate=$1
apply=${2:-}
[[ -f "$candidate" ]] || { echo "candidate report not found: $candidate" >&2; exit 2; }
[[ -z "$apply" || "$apply" == "--apply" ]] || usage

repo_root=$(git rev-parse --show-toplevel)
baseline="$repo_root/tests/baselines/retrieval-v1.json"

python3 - "$candidate" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
version = report.get("metadata", {}).get("schema_version")
if version != 2:
    raise SystemExit(
        f"candidate must use eval schema version 2, found {version!r}: {path}"
    )
if not isinstance(report.get("cases"), list) or not isinstance(report.get("summary"), dict):
    raise SystemExit(f"candidate is not an EvalRunReport: {path}")
metadata = report["metadata"]
if metadata.get("provider") != "fixture" or metadata.get("model") != "deterministic-stack-v1":
    raise SystemExit(
        "candidate must be produced by the deterministic retrieval fixture "
        "(metadata.provider='fixture', metadata.model='deterministic-stack-v1'); "
        "the CI gate cannot accept a live eval-augment report as its offline baseline"
    )
PY

echo "Reviewable baseline diff (no file has been changed):"
diff -u "$baseline" "$candidate" || true

if [[ "$apply" != "--apply" ]]; then
  echo "Baseline remains unchanged. Re-run with --apply after reviewing this diff."
  exit 0
fi

[[ -t 0 ]] || {
  echo "--apply requires an interactive terminal so a human reviews the diff." >&2
  exit 2
}
read -r -p "Type UPDATE to replace tests/baselines/retrieval-v1.json: " confirmation
[[ "$confirmation" == "UPDATE" ]] || {
  echo "Baseline remains unchanged."
  exit 1
}
cp "$candidate" "$baseline"
git diff --check -- "$baseline"
echo "Baseline updated. Inspect and commit the ordinary git diff deliberately."
