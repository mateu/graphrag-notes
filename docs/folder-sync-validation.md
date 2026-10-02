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
roots, incomplete scans, overlapping definitions, and recoverable config edits
have focused regressions. Config tests cover saves after the snapshot check,
editors recreating the config during commit, writes through an already open
file descriptor, and preservation of backup bytes and permissions. Backup
tests preserve terminal proposal audit history while requiring live proposal
endpoints and resulting edges to exist; checksum-valid malformed encoded
terminal references are rejected by both verification and restore.

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

## Live evidence

Final standalone #60 verification passed **482 workspace tests**, with 4
existing ignored tests, using all features and the locked dependency graph.
The 12 real CLI folder scenarios, 11 folder engine tests, 6 config tests,
portable ID round trips and proposal-history regressions all passed. Formatting,
whitespace checks, all-target/all-feature Clippy with warnings denied, the exact
MSRV declaration check, and 9 offline installer/source-setup checks passed.

The combined #58/#59/#60 integration at
`7fe7a49ba75fa5ab2843ca4b181a88a7723f208f` passed **493 workspace tests**,
with 5 existing ignored tests, plus formatting, whitespace checks and
all-target/all-feature Clippy with warnings denied. This combines independent
feature branches rather than adding keyword retrieval to #60.

On 2026-10-02, the combined source-built debug binary for issues #58, #59 and
#60 completed **25 checks** on macOS 27.0.1 ARM64, using existing local Ollama
and `bge-m3:latest`, with entity extraction disabled. The tested integration
commit was `66e93e78623a8f714403a4ff409cfbc5a2ddd741`, incorporating #60
commit `538b43f6b1080769895795a6d2f78cf34108d1c4`.

The walkthrough passed registration, preview, initial sync, unchanged offline
sync, provider-free keyword retrieval and inspection, manual detachment,
changed-file failure with exit 5, inspection of the last good revision,
successful resume, rejection of a superseded hit with exit 3, rename preview
and sync, confirmed prune, preservation of the manual copy, empty repeated
prune preview, vectorless backup/verification/restore, restored manual inspection
with saved IDs and redacted host URI, and an identical file returning to the
original database under its retained source ID.

The keyword checks require the independent #59 change; this PR does not
include its search implementation. The folder operations and lifecycle checks
also have standalone offline acceptance coverage. No personal corpus was
opened and existing Ollama services/models were preserved. Linux/Intel live
walkthroughs remain deferred.

Evidence was captured in temporary `graphrag-usability-integration.vg6fp0p1`
logs (`summary.json`, `steps.json`, per-command stdout/stderr) using the
`graphrag-usability-integration.py` harness. The full evidence directory was
`/private/var/folders/61/rg59jz8n7cxg8x9cdd4_9vy80000gn/T/graphrag-usability-integration.vg6fp0p1`.
