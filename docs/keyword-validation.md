# Keyword retrieval validation

Issue: [#59](https://github.com/mateu/graphrag-notes/issues/59), under the
[day-to-day usability epic](https://github.com/mateu/graphrag-notes/issues/55).

All fixtures use temporary databases, isolated HOME/configuration, and ephemeral
loopback ports. No personal corpus, running inference service, or model download
is used. The real CLI fixture seeds its database in a child process; this avoids
holding the embedded datastore's native lock/cache in the test runner while
starting the CLI against that same path. The ignored seed helper is invoked
explicitly by each fixture and is not a skipped acceptance check.

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
```

The #58 navigation baseline `a0307d3` is integrated. On macOS ARM64, the
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
