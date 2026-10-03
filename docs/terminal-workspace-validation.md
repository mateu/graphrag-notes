# Terminal workspace validation

This procedure validates [#63](https://github.com/mateu/graphrag-notes/issues/63)
against the [shared application boundary](application-boundary.md) prepared for
[#56](https://github.com/mateu/graphrag-notes/issues/56). It targets macOS and
Linux. A native Linux walkthrough and network/MCP validation are separate work.

## Isolated acceptance fixtures

`crates/cli/tests/workspace_commands.rs` starts real CLI processes with private
HOME/configuration, persistent databases, draft directories, and loopback
inference endpoints. A seed process creates the fictional corpus and exits
before a workspace opens it, releasing SurrealDB's process-scoped store registry
and RocksDB lock. No fixture opens personal notes or launches an installed
inference service or editor.

The provider double records every request and can fail or block a request.
Offline operations must make zero requests, rather than merely tolerate a
provider failure. Live subprocess tests send staged commands and SIGINT to
exercise action cancellation, recovery to the prompt, and a fresh cancellation
token on the next action. Every wait has a timeout and the process owner is
cleaned up on failure.
Accepted fixture sockets explicitly use blocking mode and bounded read/write
timeouts; an incomplete or closed cancelled request is skipped. This avoids
macOS inheriting the listener's nonblocking flag on an accepted connection.

The scenarios cover:

- Offline startup, recent notes, keyword search, selected inspection/copy,
  source status, proposals, and statistics.
- Empty corpora, unknown input, invalid selection, narrow piped output, and EOF.
- Hybrid provider failure followed by offline browsing, and recoverable quick
  capture failure without an additional note.
- Literal executable arguments and source paths containing spaces, Unicode,
  quotes, shell syntax, `#`, `%`, and ESC; filenames display with escaped
  controls while the opener receives literal arguments. A replaced result list
  clears selection.
- A competing CLI's actionable database-owner error, followed by a successful
  retry after the workspace exits.
- Cancelled read and capture actions, retained pre-commit capture input, and
  a subsequent successful capture using a fresh action token.
- Proposal skip followed by another workspace command, and the existing
  `interactive` alias with offline memory browsing and persistent-capture
  guidance.
- Cancelled proposal confirmation and EOF preserve the complete proposal card;
  confirmed accept and undo use the shared manual audit and retain review
  metadata when the resulting edge is retired.

Same-process application tests mutate a shared repository clone to check stale
selection revisions and regenerated sources. The initial terminal does not
offer source refresh/import; a separate CLI cannot mutate a corpus while its
workspace owns the database. Selection stores canonical ID plus revision and
never follows a source or title to a successor record.

## Native macOS walkthrough

`/tmp/graphrag-terminal-walkthrough.py` uses a frozen copy of the source-built binary, a fresh
temporary corpus, deterministic localhost inference, and executable editor and
opener doubles. It keeps a JSON report and command stdout/stderr in its printed
temporary evidence directory. The harness covers an empty corpus, unavailable
providers, selection without retyping IDs, literal source opening and copying,
proposal review, narrow output, competing database ownership, cancellation,
and returning to offline commands after an error.

On 2026-10-02, the Apple Silicon macOS source-binary walkthrough passed **33
checks**: **26** piped-command checks and **seven** native terminal checks. The
native pass creates a controlling PTY with **32 columns and 10 rows**. It
exercises Tab command completion, Up history recall, Down returning to a fresh
input line, Ctrl-C at an unfinished proposal confirmation, subsequent statistics
and keyword browsing, complete before/after proposal-card equality, and zero
provider requests for those actions. The editor double saves multiline content
by atomically replacing its private draft pathname; the opener records literal
arguments instead of launching a personal application.

Evidence is retained at
`/private/var/folders/61/rg59jz8n7cxg8x9cdd4_9vy80000gn/T/graphrag-terminal-macos-635y5j7d`
in `steps.json` and `summary.json`; the execution log is
`/tmp/graphrag-63-macos-walkthrough.log`. The frozen executable is
`/tmp/graphrag-63-frozen-binary` and reports the existing release-candidate
version, rather than a new published release. Its SHA-256 is
`635dfec8c7237c6c9b3084efdf29fbdcc44b86edafd0a70312897e8bbf4967b6`.

This is source-binary acceptance evidence. It does not establish native GUI
editor behavior, retrieval quality from live models, a published release
installation, native Linux behavior, or remote/MCP client interoperability.

Run the focused acceptance suite only while owning the worktree's Cargo slot:

```sh
CARGO_BUILD_JOBS=2 RUSTC_WRAPPER='' cargo test -p graphrag-cli \
  --test workspace_commands --all-features --locked --offline -- --test-threads=2
```

The focused workspace target passed **12 tests**: **11 acceptance scenarios**
and one fixture-process discovery check, with no ignored cases. The complete
workspace rerun after the accepted-socket fixture correction passed **576
tests** across **22 targets**, with **five existing cases ignored**. The final
run includes the shared safe-display fixes and the ESC filename fixture, the
12 workspace cases, and 12 shared-application cases. All-feature/all-target
Clippy passed with warnings denied, and formatting and whitespace checks passed.

Final suite and Clippy logs are `/tmp/graphrag-63-workspace-all-tests-final.log`
and `/tmp/graphrag-63-clippy-final.log`. The focused log is
`/tmp/graphrag-63-workspace-tests.log`; the native CLI build log is
`/tmp/graphrag-63-cli-build.log`.
