# Keyword retrieval validation

Issue: [#59](https://github.com/mateu/graphrag-notes/issues/59), under the
[day-to-day usability epic](https://github.com/mateu/graphrag-notes/issues/55).

All fixtures use temporary databases, isolated HOME/configuration, and ephemeral
loopback ports. No personal corpus, running inference service, or model download
is used. The real CLI fixture seeds its database in a child process; this avoids
holding the embedded datastore's native lock/cache in the test runner while
starting the CLI against that same path. The ignored seed helper is invoked
explicitly by each fixture and is not a skipped acceptance check.
When run directly with the standard `--ignored` filter without a fixture
directory, the helper returns successfully without opening a database.

| Check | Evidence |
| --- | --- |
| All hit kinds, scope/source/time filters, empty results, native BM25 weighting, deterministic ordering, limit | Two `keyword_search` integration tests passed |
| No provider methods, embedding identity/capabilities, metadata initialization, or cache records | Engine uses a provider that panics on every method; legacy vectors have no embedding metadata; cache remains empty |
| Real CLI with offline providers, all hit kinds and filters | `keyword_cli_reports_all_hit_kinds_and_consistent_filters_offline` passed; loopback listener receives zero connections |
| Explain JSON/JSONL: full-text only, BM25 score kinds, no provider identity, empty pipeline retained | `keyword_explain_jsonl_keeps_empty_pipeline_and_only_full_text_evidence` passed |
| Ordinary JSONL hit shape, mode/channels, empty stream; human empty results identify mode | `ordinary_keyword_jsonl_preserves_hit_envelopes_and_human_empty_names_mode` passed |
| Keyword search after actual backup create/restore without embeddings | `keyword_search_remains_available_after_vectorless_backup_restore` passed |
| Keyword search hints across all three hit kinds inspect the same IDs/content/revisions from another directory; source opener gets the exact original file | `keyword_navigation_hints_inspect_each_hit_kind_and_open_the_exact_source` passed against the #58 baseline |
| Option-like query/source values remain runnable in recovery hints | `hybrid_recovery_preserves_option_like_query_and_source_filter_values` passed using a query after `--` and `--source-uri=...` |
| Explicit graph-on rejected with validation exit 2 before inference | `keyword_rejects_explicit_graph_on_without_contacting_providers` passed |
| Hybrid health outage preserves exit 1; exact keyword command replays from another working directory with same config/database/filters | `hybrid_provider_failure_suggests_a_runnable_keyword_command_with_same_filters` passed |
| Hybrid health succeeds but embedding request fails; explicit recovery still offered | `hybrid_embedding_request_failure_also_offers_explicit_keyword_recovery` passed against a deterministic HTTP fixture |

Focused commands (workspace selection retains the same feature union as normal
CI while only the named integration targets execute):

```sh
RUSTC_WRAPPER='' CARGO_BUILD_JOBS=2 cargo test --workspace \
  --test keyword_search --test keyword_commands --locked -- --test-threads=2

# Direct ignored-test invocation without fixture environment also passes.
RUSTC_WRAPPER='' CARGO_BUILD_JOBS=2 cargo test -p graphrag-cli \
  --test keyword_commands --locked -- --ignored --test-threads=1
```

The #58 navigation baseline `9b68002` is integrated, including the Linux
persistent-reopen test isolation correction. On macOS ARM64, the
all-feature workspace suite passed **454 tests**, with **5 ignored** (four
pre-existing diagnostics/fixtures plus the new explicitly invoked seed helper).
This includes existing hybrid retrieval/fusion regression fixtures and all
12 navigation CLI tests. The 11 new keyword engine/CLI acceptance tests passed.
Formatting and `git diff --check` passed.
`cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`
passed without warnings.

The automated macOS CLI walkthrough uses the actual built executable with
providers unavailable, then replays printed inspect/open and hybrid-recovery
commands from another directory. It also creates and restores a real vectorless
backup. Opener behavior is tested using a fixture executable; no personal editor
or corpus is opened. Published `v0.1.0-rc.1` assets predate this feature, and no
new binary release is published by this PR.

## Live imported corpus on macOS

An additional macOS ARM64 walkthrough used the disposable #58 corpus produced
by actual Markdown and chat imports through the Librarian, including its
source-written chat metadata. With a fresh HOME/current directory, explicit
config/database paths, and both resolved inference endpoints set to
`http://127.0.0.1:1`, **13 checks passed**. The user's running Ollama service
was untouched.

`Atlas` keyword search returned 7 notes in notes scope, 1 original message in
messages scope, and 9 hits in all scope (7 notes, 1 message, 1 conversation
summary). Representative hits of every kind plus a Markdown-backed note were
inspected with revision guards. Their printed inspection commands replayed
from another directory with matching IDs, complete stored content, and
revisions. Chat UUID/context and Markdown heading/line provenance remained
intact, including the original filename's spaces, apostrophe, Unicode, `#`, and
`%`. Source-opening hints were checked for valid guarded commands and existing
original files without launching a GUI. Explain output reported only full-text
BM25 evidence, graph `Off`, and no embedding provider/model identity.
