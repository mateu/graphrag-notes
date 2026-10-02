# Capture and context validation

Validation targets macOS and Linux. All fixtures use a disposable HOME,
configuration, database, draft directory, and localhost provider endpoint.
They never read the user's corpus or require live models.

On 2026-10-02, the Apple Silicon macOS 27.0.1 run passed 18 focused CLI
subprocess tests and the full workspace suite (527 passed, 5 ignored live or
fixture-only cases). A separate macOS walkthrough passed 14 recorded command
steps, including atomic editor saves, retry from another directory, source
detach, and raw context reuse. Inference used deterministic localhost doubles.

## Automated checks

`crates/cli/tests/capture_commands.rs` runs the actual CLI with deterministic
HTTP providers and blocking executable editor doubles. It covers multiline
stdin, JSON/JSONL IDs, editor rename saves, literal argv/Unicode paths,
unchanged and cancelled sessions, private persistent recovery drafts,
option-like metadata, retry from another working directory, invalid UTF-8,
empty input, editor launch failure, offline/provider failures, source-owned
edit guidance, explicit detach, previous-note safety, post-commit cleanup
warnings, invalid UTF-8 recovery pathname rejection, symlink/non-file recovery
suppression, memory-mode rejection, and clean raw augmentation with stderr
explanations and round-trip JSON citation escaping.

The CLI unit regression passes an opening snapshot through the complete
guarded Librarian edit path after a concurrent same-timestamp change. DB
regressions verify the transaction-level current/stale/missing/hidden and
detach cases, including rollback of note and entity changes. Ten DB regressions
cover existing-entity metadata and complete entity-set rollback for rejected
snapshots and mention, note, and entity storage failures; successful duplicate
alias merging retains existing entity identity/type/creation time.

Run checks from an isolated worktree with its own target directory:

```bash
CARGO_BUILD_JOBS=2 RUSTC_WRAPPER='' cargo test --workspace --all-features --locked --offline
CARGO_BUILD_JOBS=2 RUSTC_WRAPPER='' cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo fmt --all -- --check
git diff --check
```

## macOS walkthrough

Use a temporary configuration pointing to deterministic inference HTTP doubles
with the configured embedding/extraction model names and a 1024-element
embedding. Set HOME/XDG configuration to disposable paths and pass explicit
`--config` and `--db-path` arguments. Use a blocking executable editor that
atomically replaces the passed draft pathname with multiline Markdown.

1. Capture multiline stdin with JSON output; inspect the emitted canonical ID.
2. Capture through an editor using explicit executable/argv and a pathname
   containing spaces and Unicode. Confirm editor diagnostics stay on stderr.
3. Open a manual note in an unchanged editor with title/tags/detach flags.
   Confirm the original note is unchanged and inference receives no requests.
4. Make an editor write content and exit nonzero. Confirm cancellation,
   private retained draft, and a copyable retry preserving config/database.
5. Fail extraction during a changed edit. Confirm the old note is intact;
   restore the fixture and run the printed recovery command.
6. Import temporary Markdown. Confirm an in-place editor attempt offers
   open/detach guidance; explicitly detach a changed body into a manual note.
7. Run raw augment with explain and separate stdout/stderr files. Confirm
   only prompt content and stable citations appear in stdout. Check an empty
   context and zero budget produce empty stdout.

The focused subprocess suite executes these editor and provider flows on the
macOS host. Native GUI editor behavior and Linux walkthroughs remain separate
environment checks; no GUI application or installed inference service is
changed by this validation.
