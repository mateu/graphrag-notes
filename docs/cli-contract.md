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
