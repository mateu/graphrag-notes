#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_PATH="${DB_PATH:-/tmp/graphrag-phase7-validation}"
EXPORT_PATH="${EXPORT_PATH:-$ROOT_DIR/conversations-7.json}"
EVAL_PATH="${EVAL_PATH:-$ROOT_DIR/tests/augment-eval-sample.jsonl}"

if [[ ! -f "$EXPORT_PATH" ]]; then
  echo "Missing export file: $EXPORT_PATH" >&2
  exit 1
fi

if [[ ! -f "$EVAL_PATH" ]]; then
  echo "Missing eval file: $EVAL_PATH" >&2
  exit 1
fi

cd "$ROOT_DIR"

print_step() {
  echo
  echo "==> $1"
}

run_timed() {
  local label="$1"
  shift
  local start end duration
  start="$(date +%s)"
  "$@"
  end="$(date +%s)"
  duration="$((end - start))"
  printf "   [%s] %ss\n" "$label" "$duration"
}

print_step "Reset DB"
run_timed "reset-db" cargo run -q -p graphrag-cli -- reset-db --db-path "$DB_PATH" || true

print_step "Import (hybrid, skip extraction)"
run_timed "import" cargo run -q -p graphrag-cli -- --db-path "$DB_PATH" import-chats "$EXPORT_PATH" --mode hybrid --skip-extraction

print_step "Stats after import"
cargo run -q -p graphrag-cli -- --db-path "$DB_PATH" stats

print_step "Migrate dry-run"
run_timed "migrate-dry" cargo run -q -p graphrag-cli -- --db-path "$DB_PATH" migrate-chats "$EXPORT_PATH" --dry-run

print_step "Migrate records-only"
run_timed "migrate" cargo run -q -p graphrag-cli -- --db-path "$DB_PATH" migrate-chats "$EXPORT_PATH"

print_step "Stats after migrate"
cargo run -q -p graphrag-cli -- --db-path "$DB_PATH" stats

print_step "Search smoke"
run_timed "search-notes" cargo run -q -p graphrag-cli -- --db-path "$DB_PATH" search "navidrome" --scope notes --limit 3
run_timed "search-messages" cargo run -q -p graphrag-cli -- --db-path "$DB_PATH" search "turtle beach dongle" --scope messages --limit 3
run_timed "search-all" cargo run -q -p graphrag-cli -- --db-path "$DB_PATH" search "homelab rtx 5060" --scope all --limit 5

print_step "Augment smoke"
run_timed "augment-all" cargo run -q -p graphrag-cli -- --db-path "$DB_PATH" augment "navidrome self-hosted music server" --scope all --limit 5 --max-tokens 300 --max-chunk-tokens 80
run_timed "augment-notes" cargo run -q -p graphrag-cli -- --db-path "$DB_PATH" augment "homelab gpu power" --scope notes --limit 5 --max-tokens 300 --max-chunk-tokens 80

print_step "Eval harness"
run_timed "eval-augment" cargo run -q -p graphrag-cli -- --db-path "$DB_PATH" eval-augment "$EVAL_PATH" --fail-on-miss

print_step "Validation complete"
echo "All validation steps passed."
echo "DB path: $DB_PATH"
