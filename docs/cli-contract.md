# CLI output and safety contract

The modular command surface introduced in v0.2 uses `--format human|json|jsonl`.
Human remains the default for interactive use. JSON emits one versioned envelope;
JSONL emits one envelope per listed record so it can be streamed safely.

```json
{
  "schema_version": 1,
  "command": "notes.show",
  "success": true,
  "data": { "id": "note:example" },
  "warnings": [],
  "errors": []
}
```

Only requested command data is written to stdout. Logs, provider diagnostics,
deprecation notices, and progress belong on stderr.

| Exit code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Internal failure |
| 2 | Validation or unsafe invocation |
| 3 | Requested record not found |
| 4 | Embedding/model compatibility failure |
| 5 | Partial durable-processing failure |

`import-chats` and `migrate-chats` exit 5 when any conversation fails. Successful
peers remain usable. Input totals, conversation type counts, and `messages_total`
include failed conversations. Creation, upsert, link, and Q&A outcome counters
cover completed conversations; partial writes from failed conversations may be
omitted from those counters. Failure guidance is written to stderr. Fix the
reported error and retry the same export.

`init` supports `--format human|json`. Its default preview and explicit
`--write` do not open the database or contact providers. Adding `--check`
returns provider diagnostics in the setup report: exit 0 means healthy, 1 means
warnings, and 2 means a failed check. JSON retains the setup data on diagnostic
failure, sets `success` to false, and includes warning/error summaries.

## Notes commands

`graphrag notes list [--tag TAG] [--source-uri URI]` lists visible notes.
`notes show ID` returns the complete note. `notes edit ID` accepts metadata
changes plus `--content-file PATH` or explicit `--stdin`; changing content
re-embeds and re-extracts before replacing the persisted note. A provider
failure therefore leaves the old searchable note untouched.

Source-generated notes cannot be edited in place. `notes edit ID --detach`
creates a new manual note; it retains the original `source_id` as provenance
but has no source generation, so reimport cannot overwrite it.

`notes delete ID` is a safe preview unless `--yes` is supplied. `--dry-run`
always previews. The output reports the exact affected mentions, accepted
edges, mutable proposals, and chat provenance; deletion never removes the
source record or unrelated notes.

The top-level `list` and `show-note` commands remain supported in v0.2 and
emit a stderr deprecation message directing callers to `notes list` and
`notes show`.

## Search-to-source navigation

Human search results show a readable title, hit kind, a query-centered preview,
provenance, and an exact inspection command. File-backed results also show an
explicit opening command. These additions do not change retrieval ranking or
replace the stored full text with its preview.

Search JSON and JSONL keep their existing result fields and add `navigation`:

- `title` and `preview` provide the readable result label and a bounded Unicode
  preview centered on query terms when a match is available.
- `inspect_command` and optional `open_command` include the canonical record ID,
  database selection, and revision guard for that displayed snapshot.
- `revision` is an opaque record revision token.
- `provenance` retains source identity, heading/line context when known, and
  conversation/message identity for chat hits.
- `warning` explains an unavailable or changed snapshot when applicable.

If the record changes or becomes unavailable while search output is enriched,
navigation can have null revision/provenance and a warning. Its inspection
command retains a guard that fails safely rather than selecting newer content.

`--explain` retains its existing evidence and pipeline fields and adds the same
navigation information. Consumers should use canonical IDs and revision tokens,
and treat the printed commands as user-facing conveniences rather than execute
them automatically. A `file://` source URI identifies the stored source; it does
not imply that a remote client's computer has that file.

This change upgrades the database to schema 15 so chat metadata can retain its
provider fields. Existing notes and chat records are preserved. Take a backup
before opening an existing corpus with the new binary: older binaries,
including v0.1.0-rc.1, reject databases with this newer schema. Downgrading
requires restoring the earlier backup with the matching binary.

## Inspect a result

```bash
graphrag inspect note:ID
graphrag inspect message:ID --neighbors 2 --format json
graphrag inspect conversation:ID --format jsonl
graphrag inspect note:ID --revision TOKEN
```

`inspect` accepts full `note:`, `message:`, or `conversation:` record IDs. Search
row numbers are display labels; `inspect 1` and `open 1` are rejected. Inspection
uses the local database and contacts no inference providers.

Its versioned envelope has `command: "inspect"`. `data` contains `id`,
`hit_type` (`note`, `message`, or `conversation-summary`), `title`, complete
`content`, `revision`, `provenance`, `conversations`, `messages`,
`messages_truncated`, and `warnings`. JSONL emits one envelope for the inspected
record. Provenance includes source ID/URI/type/generation and heading/line range
when available; chat provenance includes conversation ID/UUID, original
`message_uuid`, `message_key`, message index, and role. Nested messages also
retain their original UUID/key alongside the canonical `message:ID`. For
exports without a message UUID, `message_uuid` is null and `message_key` is the
importer's conversation-UUID/index fallback. Human search and inspection show
the original UUID or fallback key. Message indices remain zero-based in machine
output.

For a message, `--neighbors` includes that many messages before and after the
selected message, plus the selected message itself. The default is 2 and the
maximum is 20. Conversation and derived-note inspection also return bounded
message context, with `messages_truncated` when more context is available.
Nested message and conversation revision tokens are usable with direct
inspection of those IDs.

The commands printed by search include `--revision` so a changed or reused
record cannot silently replace the displayed selection. A mismatched revision
exits 2; a missing or superseded record exits 3. Search again to choose a current
result. Omitting the guard deliberately inspects the record's current state.

## Open an original local file

```bash
graphrag open note:ID --revision TOKEN
graphrag open message:ID --opener /absolute/path/to/editor --format json
```

`open` is an explicit action that opens the original local file. It never opens
anything automatically while searching or inspecting, contacts no inference
providers, and does not reimport or edit stored notes. Missing files exit 3 with
recovery guidance; stored note text remains inspectable. Records without a local
file source and non-file URIs are rejected with exit 2. A missing or stale ID
cannot select another result by its row number.

The opener selection order is explicit `--opener PATH`, resolved
`[navigation].opener` (including `GRAPHRAG_OPENER`), `VISUAL`, `EDITOR`, then the
platform default. CLI and environment values are executable paths, each passed
as one value; they are not shell command strings. To include fixed editor
arguments, configure an argv array:

```toml
[navigation]
opener = ["/absolute/path/to/editor", "--reuse-window"]
```

The CLI appends the original file path as one literal argument. It never runs
an opener through a shell or expands command substitutions, variables, quotes,
or wildcard characters in that path. Source URIs preserve GraphRAG's literal
file-path format, including `#` and `%` characters.

Successful machine output has `command: "open"` and `data` containing `id`,
`path`, `launched`, opener argv, `provenance`, and `revision`. `launched: true`
reports that the configured opener succeeded; it does not confirm that a GUI
editor finished loading the document. Opening uses the same revision guard as
inspection and validates it before launching the opener.

## Keyword retrieval

`search --mode hybrid|keyword` defaults to hybrid. The selected mode and base
channels are additive machine metadata; see [keyword search](keyword-search.md)
for the JSON/JSONL field contracts, filters, ranking, and provider-free behavior.
Hybrid failures preserve their exit code and print a copyable explicit keyword
command on stderr. The CLI does not silently change retrieval mode.

## Connection review

`garden review` is a provider-free inbox for persisted proposals. The default
filter is pending; `--id proposed_edge:ID`, `--status`, and `--all-statuses`
select an exact proposal or lifecycle states. `--limit` is bounded to 1–200.
Human cards show both notes, bounded excerpts, provenance, confidence, reason,
proposal state, audit timestamps/manual flags, and available undo. Printed
follow-up commands preserve the selected config and database. A non-UTF-8
replay path makes `review_command`, `inspect_command`, and `undo_command` null
with card `warnings`; cards and interactive decisions remain available without
lossy replacement hints. See [connection review](connection-review.md).

Read-only JSON uses `command: "garden.review"` and `data.proposals`; JSONL emits
one envelope per proposal with its card in `data`, and an empty inbox emits no
lines. Interactive review requires human format; prompts and confirmations use
stderr. Accept/reject/undo require a matching typed confirmation followed by an
optional audit reason. Skip, cancellation, EOF, and quit do not write a decision.
Existing `garden proposals accept --all` still requires `--min-confidence` and
`--yes` and retains its Gardener/related-to restrictions.
