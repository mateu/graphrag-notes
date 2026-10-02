# Folder sync validation

This procedure covers [issue #60](https://github.com/mateu/graphrag-notes/issues/60).
Use a binary containing the folder commands and isolated temporary paths.
Published candidates may predate this feature. Keep existing notes and running
providers untouched.

## Automated checks

```bash
cargo test --locked -p graphrag-config folder
cargo test --locked -p graphrag-agents ingestion::folders::tests
cargo test --locked -p graphrag-cli --test folder_sync_commands
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
```

The engine tests use deterministic inference doubles and temporary folders.
The CLI tests run real subprocesses against temporary persistent databases,
cleared environments, and isolated local HTTP fixtures. Request logs verify
that registration, previews, unchanged sync, prune, and portable backup/restore
do not call providers. No fixture contacts a live inference service or opens
a personal corpus.

Coverage includes failed changed files retaining their searchable generation,
successful peers, checkpointed cancellation, deferred generation cleanup,
pinned roots/rules/identities, repaired unreadable files, exact copied retry
commands from another working directory, renames, current prune tokens,
manual/legacy/detached notes, source identity/content through vectorless backup
and restore with host-local URI redaction, and identical files returning to
the original database reusing the retained source ID. Symlink
cycles, nested directory replacements, redirected root ancestors, unavailable
roots, incomplete scans, overlapping definitions, and atomic config edits
have focused regressions. Backup tests preserve terminal proposal audit
history while requiring live proposal endpoints and resulting edges to exist.

## Manual local walkthrough

Use already configured local inference models for changed imports. Inspect
the provider settings before running them. This setup uses copied fictional
notes and an explicit database/config:

```bash
folder_check_dir="$(mktemp -d "${TMPDIR:-/tmp}/graphrag-folders.XXXXXX")"
mkdir "$folder_check_dir/notes"
cp samples/first-notes.md "$folder_check_dir/notes/first-notes.md"
./target/debug/graphrag --config "$folder_check_dir/config.toml" \
  --db-path "$folder_check_dir/data" init --backend ollama --write
./target/debug/graphrag --config "$folder_check_dir/config.toml" \
  folders add work "$folder_check_dir/notes"
./target/debug/graphrag --config "$folder_check_dir/config.toml" \
  sync work --dry-run --format json
./target/debug/graphrag --config "$folder_check_dir/config.toml" sync work
```

1. Repeat sync with an unavailable embedding endpoint override. Confirm the
   result is unchanged, no job is created, and note IDs/generations remain.
2. Change the copied Markdown while that override is active. Confirm exit 5,
   the old stored note remains visible, and failed-file/job hints are usable.
3. Clear the override and copy the exact resume command. Confirm replacement
   succeeds under the pinned source ID and unchanged peers make no inference
   requests. Newly created files wait for a fresh named-folder sync.
4. Create a detached manual copy using `notes edit ID --detach`. Rename the
   Markdown and sync again. Confirm the new path imports and the old path is
   reported missing while both the old stored generation and manual copy remain.
5. Preview `folders prune work --format json`. Confirm stale tokens refuse
   deletion; then copy its current confirmation command. The old generated
   notes should disappear, while manual content and source provenance remain.
6. Create and verify a default vectorless portable backup. Restore it into a
   fresh explicit `--db-path` and inspect the manual note/source using their
   saved IDs. Host-local URIs should be redacted. Return to the original
   database, restore the original file, and confirm sync regenerates notes
   under the same source ID.
7. Record platform, binary version/commit, provider/model identities, observed
   commands/results, and failure/recovery behavior. Live keyword retrieval can
   additionally be checked when the independent issue #59 feature is present.

Use the same selected config/database in every follow-up command. If the
source binary is not on PATH, replace only the leading `graphrag` executable
in copied hints with its absolute path. Automated fixtures establish lifecycle
and CLI behavior; they do not establish live-model quality or another platform's
behavior.
