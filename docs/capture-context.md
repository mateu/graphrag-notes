# Capture, revise, and reuse context

These commands are available in current source and the forthcoming Apple
Silicon `0.1.0-rc.2` candidate. Use source until its assets are published;
`v0.1.0-rc.1` predates them. [Getting started](getting-started.md) covers
installation and providers; the [daily workflow](daily-workflow.md) connects
capture to refresh, retrieval, review, and recovery. Native Linux and Intel
macOS release acceptance remains in [#66](https://github.com/mateu/graphrag-notes/issues/66).

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
manual note and atomically stores its entities and mentions after embedding and
required extraction succeed. It creates no source record. A provider failure
creates no note or partial source; the input remains in a recovery draft.
The existing `add` command and its output keep their previous behavior.
Recoverable `capture` and note content/editor edits require a persistent
database. `--memory` is rejected before input, editor launch, or database work
because a later recovery process cannot reopen the same ephemeral corpus.
Other legacy commands keep their existing memory behavior.

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
no `source_generation` or chat ownership relationships, so subsequent manual
edits remain allowed even when they keep the original chat tags and source
type. Chat-import ownership uses stored `note_from_conversation` and
`note_from_message` relationships, including when the linked chat record is
missing. New chat notes and their conversation ownership commit together,
before entity extraction or message-specific links. Interrupted imports
therefore keep visible chat notes protected. Refused chat edits print an
`inspect` command and explicit detach command before any editor or inference work.

Historical chat imports that have neither generation ownership nor stored
chat relationships cannot be distinguished automatically from old detached
manual copies. Those unlinked legacy records remain editable; current chat
imports and existing linked imports are protected.

## Recover a draft

GraphRAG saves a persistent draft before launching an editor or checking
providers. By default the directory is the full database pathname with
`.drafts` appended, such as `knowledge.surreal.drafts`. Use `--draft-dir PATH`
to choose another location. Newly created directories use permissions 0700
and draft files use 0600 on macOS/Linux. Files saved by editor rename are
read from their pathname and returned to private permissions after exit.
The editor must leave a regular file, not a symlink.
If it leaves a symlink, directory, or missing pathname, GraphRAG reports the
rejected path and withholds all recovery commands. Inspect the path, remove
the replacement if needed, and restore the intended draft before retrying.

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
Unchanged editor sessions also attempt cleanup before printing their result.
If cleanup fails, JSON/JSONL keeps `status: "unchanged"`, includes the retained
`draft_path`, and leaves `recovery_command` null. No note was created or
changed; inspect/remove the leftover working draft when convenient.

## Reuse clean prompt context

```bash
graphrag augment "Atlas launch plan" --raw > atlas-context.txt
graphrag augment "Atlas launch plan" --raw --explain > atlas-context.txt
```

Raw stdout contains only the existing packed prompt block and its `[C1]`,
`[C2]`, … citation dictionary with canonical IDs and available source/chat
provenance. It omits query headings, scores, progress, and packing summaries.
Each dictionary record occupies one line, for example
`[C1] id="note:ID", source_uri="file:///path/notes.md"`. All string values
(`id`, `source_uri`, and `conversation_uuid`) use JSON string quoting:
newlines, carriage returns, and tabs become `\n`, `\r`, and `\t`; quotes and
backslashes become `\"` and `\\`; remaining control characters use JSON
escapes such as `\u001b`. Unicode controls and line separators are also escaped,
including `\u007f`, `\u0085`, `\u2028`, and `\u2029`. Decoding each quoted
value as a JSON string recovers the exact original value. `message_index`
remains a one-based numeric value.
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
