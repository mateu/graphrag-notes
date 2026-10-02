# Capture, revise, and reuse context

These commands are available in the latest source build. The published
`v0.1.0-rc.1` candidate predates them. [Getting started](getting-started.md)
covers installation and providers; this guide covers daily macOS and Linux use.

## Capture multiline notes

Paste or type a multiline note, then press Ctrl-D on an empty line:

```bash
graphrag capture --title "Atlas decisions" --tags atlas,decisions
```

Omitting `CONTENT` reads stdin; `--stdin` makes that choice explicit. Piped
content, literal `CONTENT`, and `--content-file PATH` are also supported:

```bash
graphrag capture --stdin --format json <<'NOTE'
Atlas launch decisions

- Finish the onboarding guide before launch.
- Review feedback on Friday.
NOTE
graphrag capture "Review the launch checklist" --tags atlas
graphrag capture --content-file ./meeting.md
```

A successful capture prints the canonical `note:ID` and an inspection command
that preserves the selected configuration and database. Capture creates a
manual note and atomically stores its entity mentions after embedding and
required extraction succeed. It creates no source record. A provider failure
creates no note or partial source; the input remains in a recovery draft.
The existing `add` command and its output keep their previous behavior.

## Use a blocking editor

```bash
graphrag capture --editor --title "Atlas decisions" --tags atlas
graphrag notes edit note:ID --editor
graphrag notes edit note:ID --editor --editor-command code --editor-arg=--wait
```

Capture starts with a blank draft, or with literal `CONTENT` when supplied.
Note editing starts with the current full note body. Selection uses
`--editor-command`, then `VISUAL`, then `EDITOR`, then `vi`. Each value is one
literal executable path. `EDITOR="code --wait"` does not split into a command
and arguments; use `--editor-command code --editor-arg=--wait`. Repeat
`--editor-arg` for additional literal arguments. GraphRAG appends the draft
pathname as one argument and waits for the editor to exit. Paths with spaces,
Unicode, quotes, or shell metacharacters do not trigger shell expansion.
Choose an editor that waits for editing to finish; GUI editors often need
an explicit wait flag. Editor output goes to stderr so JSON stays parseable.

Exiting successfully with unchanged bytes is a no-op. Exiting with a nonzero
status is cancellation: no note changes, and the draft is retained even if
the editor wrote content before exiting. Both return exit 0 with an explicit
status. Explicit `--title`, `--tags`, and `--detach` are also ignored in an
unchanged or cancelled editor session. To apply metadata alone, use the
existing `notes edit note:ID --title TITLE --tags TAGS` without `--editor`.

Changed manual content is re-embedded and re-extracted before replacement.
An editor session retains the exact opening note snapshot. Its final database
transaction rejects a note that changed, disappeared, or became hidden while
editing or during provider work. This comparison covers the complete stored
note, including content changes with an unchanged timestamp. A conflict exits
2 and retains the draft; inspect the current note and reconcile the draft
before deliberately retrying. Existing non-editor operations retain their
previous behavior.

## Respect imported sources

Source-generated chunks cannot be edited in place. An attempted edit prints
copyable commands to open the original file or explicitly detach a manual
copy, before launching an editor or contacting providers:

```bash
graphrag open note:ID
graphrag notes edit note:ID --detach --editor
```

Opening reuses the existing local source and opener validation. Edit the
original Markdown file, then sync its registered folder or reimport it.
Chat-derived records without a local file can be inspected or detached;
they cannot be opened as local files. A changed `--detach --editor` session
creates a new manual note with a new ID, preserving source provenance. The
generated original is untouched. Detached notes retain `source_id` but have
no `source_generation`, so subsequent manual edits remain allowed.

## Recover a draft

GraphRAG saves a persistent draft before launching an editor or checking
providers. By default the directory is the full database pathname with
`.drafts` appended, such as `knowledge.surreal.drafts`. Use `--draft-dir PATH`
to choose another location. Newly created directories use permissions 0700
and draft files use 0600 on macOS/Linux. Files saved by editor rename are
read from their pathname and returned to private permissions after exit.
The editor must leave a regular file, not a symlink.

Empty content, invalid UTF-8, editor launch failures, provider failures,
storage failures, and stale editor conflicts retain the draft. Invalid
UTF-8 bytes remain available for repair. Stderr prints its absolute path
and an exact recovery command, for example:

```text
Draft retained: /path/knowledge.surreal.drafts/graphrag-EXAMPLE.md
Recover: graphrag --config '/path/config.toml' --db-path '/path/knowledge.surreal' capture --content-file '/path/knowledge.surreal.drafts/graphrag-EXAMPLE.md' --title='Atlas decisions' --tags='atlas' --format json
```

The command preserves explicit configuration, database, metadata, detach
choice, draft directory, and output format. Paths for configuration, database,
and recovery content are absolute; it can be pasted from another directory.
Configuration, database, and draft pathnames must be valid UTF-8 so printed
recovery commands preserve their exact names; invalid paths are rejected
before capture input or editor launch.
Repair the draft or restore provider access first. Note-edit recovery uses
`notes edit ID --content-file PATH`; it deliberately applies the reconciled
content to the current note. A successful retry removes its working copy;
the original input draft remains until you remove it.

Successful capture/edit removes its working draft. Cleanup failure is a
warning after success: stderr identifies the successful note ID and retained
path, and capture JSON includes that path with no recovery command. Inspect
the saved note and remove the leftover draft; retrying that capture could
create another note. A draft cannot be retained if an editor itself deletes
it or input cannot be read in the first place.

## Reuse clean prompt context

```bash
graphrag augment "Atlas launch plan" --raw > atlas-context.txt
graphrag augment "Atlas launch plan" --raw --explain > atlas-context.txt
```

Raw stdout contains only the existing packed prompt block and its `[C1]`,
`[C2]`, … citation dictionary with canonical IDs and available source/chat
provenance. It omits query headings, scores, progress, and packing summaries.
`--explain` writes diagnostic details to stderr. No selected context produces
empty stdout. Existing scope, source/entity filters, graph selection, deduping,
packing, and token budgets remain available. Budgets bound the existing
prompt block; the appended citation dictionary is additional provenance.
Without explain, a zero budget skips inference and emits no raw text.

`--raw` conflicts with any explicit `--format`, including `--format human`.
Use existing JSON/JSONL augmentation for structured API data. Raw context can
be pasted into OpenClaw or Hermes today; editor interaction belongs in a local
terminal. A network/MCP interface remains the epic's separate follow-up.

## Versioned capture output

`capture --format json` emits one schema-version-1 envelope with command
`capture`. JSONL emits the same one-line envelope. The `data` fields are:

| Field | Meaning |
| --- | --- |
| `status` | `created`, `unchanged`, or `cancelled` |
| `id` | Canonical new note ID for `created`; otherwise null |
| `draft_path` | Retained draft pathname on cancellation or cleanup warning; otherwise null |
| `recovery_command` | Copyable command on cancellation; otherwise null |

An unchanged/cancelled `notes edit --editor` envelope uses command `notes.edit`
with `status`, the existing `id`, `detached: false`, `draft_path`, and
`recovery_command`. Successful changed edits preserve the existing `note`
and `detached` output shape. The shared output schema and folder-sync schema
stay at version 1. Validation exits 2, provider/storage/launch failures exit
1, and the existing missing-record and compatibility exit codes still apply.
Failures write recovery/error information to stderr without success data on
stdout. See [validation evidence](capture-validation.md) and the
[CLI contract](cli-contract.md).
