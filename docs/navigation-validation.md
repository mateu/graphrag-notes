# Search-to-source validation

This procedure covers [issue #58](https://github.com/mateu/graphrag-notes/issues/58).
Use a source binary containing the navigation change and isolated temporary
paths. Existing release candidates may predate these commands.

## Automated verification

`crates/cli/tests/navigation_commands.rs` seeds temporary persistent databases
through the repository API in isolated helper processes. Each helper exits
before the real CLI opens the database, releasing the embedded store's
process-scoped RocksDB lock. CLI invocations use a cleared environment. A
listening endpoint records whether inspection or opening unexpectedly contacts
an inference provider. A fake opener records literal arguments and emits
diagnostic output to verify machine output remains parseable. No test launches
a user's editor or accesses a personal database.

The targeted scenarios cover:

- Complete Unicode note content, source identity/generation, headings/lines,
  and versioned JSON/JSONL envelopes.
- Message and conversation inspection with bounded neighboring messages,
  original IDs/UUIDs/roles/indices, and usable nested revision tokens.
- Derived-note chat provenance, canonical-ID validation, and neighbor limits.
- Changed-record revision guards and superseded IDs after source refresh.
- Explicit opening, missing/deleted sources, and manual/non-file refusal.
- Executable paths with spaces, literal shell metacharacters in filenames and
  configured arguments, and opener precedence.

Query-centered preview behavior and Unicode boundaries are covered by focused
renderer tests. These checks validate navigation and safety, rather than
live-model retrieval quality.

Run the targeted tests after compiling the current change:

```bash
cargo test --locked -p graphrag-cli --test navigation_commands
```

## Manual local walkthrough

Use the already configured Ollama models for import/search; inspection and
opening remain available when inference services are unavailable. These
commands create a separate config and database rather than use personal notes:

```bash
navigation_check_dir="$(mktemp -d "${TMPDIR:-/tmp}/graphrag-navigation.XXXXXX")"
cp samples/first-notes.md "$navigation_check_dir/first-notes.md"
./target/debug/graphrag --config "$navigation_check_dir/config.toml" \
  --db-path "$navigation_check_dir/data" init --backend ollama --write
./target/debug/graphrag --config "$navigation_check_dir/config.toml" \
  import "$navigation_check_dir/first-notes.md"
./target/debug/graphrag --config "$navigation_check_dir/config.toml" \
  search "What is the Atlas project launch plan?" --limit 3
```

1. Confirm each result shows its kind, readable title, useful preview, original
   source, heading/line context when present, and exact inspection command.
2. Copy the printed inspection command. Confirm it returns the complete note
   and preserves the result ID and source provenance. Check JSON and JSONL too.
3. Copy the printed opening command. Confirm it opens the copied Markdown file
   through the selected editor/opener and leaves stored notes unchanged.
4. Confirm `inspect 1` rejects a row number. Change and reimport the copied
   source, then retry the previous inspection command; it must reject the stale
   ID or revision and direct you to a fresh search.
5. Move the copied file aside. Inspection should still return stored content;
   opening should report the missing file with recovery guidance.
6. For a sanitized chat fixture, inspect a message ID and conversation ID.
   Verify roles and original message order, bounded neighboring context, and
   JSON conversation/message identities.
7. Record OS/architecture, binary commit/version, provider/model identities,
   observed commands/results, and failure/recovery behavior below.

The printed commands select the resolved database path and carry revision
guards. If using the source binary without installing it on PATH, replace only
the leading `graphrag` executable with its absolute source-binary path.

## Evidence

| Scenario | Status | Evidence |
| --- | --- | --- |
| Workspace regression checks | Passed | `cargo test --workspace --all-features --locked` passed 443 tests with 4 existing ignored provider tests. `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`, formatting, and whitespace checks passed. |
| Offline CLI navigation scenarios | Passed | On macOS Apple Silicon, `cargo test -p graphrag-cli --locked --test navigation_commands` passed 12 tests: 11 acceptance scenarios and the isolated fixture-process helper, with 0 failures/ignored. No provider requests were received. |
| Unicode/query-centered preview checks | Passed | The CLI unit suite passed 83 tests, including the late Unicode query preview and literal file-URI/shell-hint checks. |
| Real chat import and partial failure | Passed | Deterministic Ollama CLI fixtures import actual chat metadata, search and inspect every chat hit kind, and verify failed conversations exit 5 while successful peers and already-durable partial records remain inspectable. |
| Manual macOS navigation walkthrough | Passed | On 2026-10-02, macOS 27.0.1 ARM64, a source-built debug binary reporting `graphrag 0.1.0-rc.1` completed 20 checks with local Ollama and `bge-m3:latest`. An isolated config/database and copied sample validated human/JSON/JSONL search and inspection, the copied command from another working directory, inspection with unavailable provider endpoints, explicit `/usr/bin/open -g -t`, Unicode and literal `#`, `%`, and quote characters in the filename, ordinal rejection (2), missing-file recovery (3), stale source-generation rejection (3), successful reimport, chat import/backfill, all three search hit kinds, message/conversation context, and healthy doctor diagnostics. Entity extraction was disabled for this navigation walkthrough. |
| Linux/Intel walkthrough | Deferred | Broader platform validation follows the macOS work. |

Record observed results only after running each check. Passing deterministic
fixtures does not establish live-model quality or another platform's behavior.

The live walkthrough used the working tree containing this change, rather than
the published candidate asset. Evidence was captured in temporary
`graphrag-navigation-live.3zwl0zdo` logs (`steps.json`, per-command stdout/stderr,
and `summary.json`). No personal corpus was opened. Schema 15 fixes the actual
chat metadata persistence failures encountered during the walkthrough; an
existing-schema upgrade test verifies prior records survive that migration.
