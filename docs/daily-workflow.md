# A repeatable daily workflow

Use current source for this workflow. The forthcoming `0.1.0-rc.2` Apple Silicon
candidate collects these commands; its assets are not published yet. The
published `rc.1` supports the older onboarding path. See [getting started](getting-started.md)
and the [candidate release guide](releases/0.1.0-rc.2.md) for installation.

For normal use, keep your usual selected configuration and database. The
practice run below creates a separate fictional Atlas notebook and leaves
personal notes untouched. Run it after building source and configuring the
local Ollama models described in getting started.

## Prepare the practice notebook

```sh
DEMO=$(mktemp -d "${TMPDIR:-/tmp}/graphrag-daily.XXXXXX")
printf '%s\n' "$DEMO"
mkdir -p "$DEMO/notes"
cd "$DEMO"
gn() { graphrag --config "$DEMO/config.toml" --db-path "$DEMO/database" "$@"; }
gn init --backend ollama
gn init --backend ollama --write
gn init --check
```

Keep this terminal open: `gn` supplies the same explicit paths to each command.
The configuration preview shows effective provider settings. If you have
exported provider or model overrides, check that it still selects the intended
local Ollama endpoint and 1024-dimension embedding model. The practice directory
remains available after the run; its printed path is in `DEMO`.

## Capture and refresh

Capture a manual note, or choose a blocking editor for a longer thought:

```sh
gn capture "Atlas pilot review is Friday; keep participant feedback together." \
  --title "Atlas pilot review" --tags atlas,pilot
gn capture --editor --title "Atlas decisions" --tags atlas
```

The editor command waits for the editor to finish. Use `--editor-command code
--editor-arg=--wait` for an editor that needs a wait flag. Capture prints the
saved ID and an inspection command. Provider or editor failures retain a
private draft and print an exact recovery command. See [capture and context](capture-context.md).

Create a Markdown source and register its folder once:

```sh
cat > "$DEMO/notes/atlas.md" <<'NOTE'
# Atlas launch plan

Atlas is a fictional notebook project. First test with the internal team,
then invite a small pilot, then open access after addressing pilot feedback.
The launch requires a working setup guide and links back to original notes.
NOTE
gn folders add work "$DEMO/notes"
gn sync work --dry-run
gn sync work
```

Edit the original file when the plan changes, then preview and apply the refresh:

```sh
printf '\n## Pilot follow-up\nReview pilot feedback before opening access.\n' \
  >> "$DEMO/notes/atlas.md"
gn sync work --dry-run
gn sync work
gn sync work
```

The final unchanged sync makes no inference requests and creates no new job.
A failed refresh keeps the last successful content searchable. Missing files
remain searchable until a separate prune preview and revision confirmation.
See [folder sync](folder-sync.md) for rules, cancellation, and prune ownership.

## Find, inspect, and open

Use keyword search when providers are stopped, or hybrid search for semantic retrieval:

```sh
gn search "launch" --mode keyword --scope all
gn search "What must happen before opening the Atlas pilot?" --limit 3
```

Standalone search prints copyable **Inspect** and **Open source** commands.
They preserve the selected database, full record ID, and revision. A changed
result requires a fresh search. Local-file opening needs an available original
file and launches only the configured literal executable arguments.

For repeated browsing, select once inside the terminal workspace:

```sh
gn workspace
```

```text
keyword launch
select 1
inspect
open
copy
back
status
quit
```

Choose a result with a local file source, using its displayed number in place
of `1`. The number refers to the current result list. The workspace keeps its
ID and revision, so `inspect`, `open`, and `copy` do not require retyping an ID.
`copy` prints selected content; it does not write the system clipboard. Quit
before running another command against that database. See [terminal workspace](terminal-workspace.md).

## Reuse context and review connections

```sh
gn augment "Atlas launch plan and pilot feedback" --raw > "$DEMO/atlas-context.txt"
gn garden scan
gn garden review --interactive
```

The context file contains packed source text and citations for reuse in a
prompt, including OpenClaw or Hermes. `augment` uses inference; keyword search
and selected workspace copying work offline. No matching context produces
empty raw output. Keep the citations with the text when you reuse it.

A scan proposes connections according to the configured similarity policy;
it may produce an empty inbox. Review shows both endpoints and provenance.
Accept, reject, and undo require deliberate confirmation; skip writes no
decision. Saved proposals remain reviewable offline. See [connection review](connection-review.md).

## Recover and preserve the notebook

| Situation | Next action |
| --- | --- |
| Capture or editor failed | Repair the retained draft or restore provider access, then copy the printed recovery command. |
| Changed-file sync partly failed | Read the per-file report; copy its retry command. `sync --resume JOB` uses the job's pinned roots and file identities. |
| Sync cancelled with Ctrl-C | Resume the printed job after the owning process exits. Pending files have not been attempted. |
| Hybrid search cannot reach a provider | Copy its explicit keyword fallback command, or run `gn search "launch" --mode keyword`. |
| A search result changed | Search again and use the new ID/revision command. |
| Database is in use | Quit the owning workspace or wait for the current CLI operation to finish, then retry. |

Create and verify a portable backup before an upgrade:

```sh
gn backup create "$DEMO/backup" --include-embeddings
gn backup verify "$DEMO/backup"
gn doctor
```

Including embeddings requires recorded model identity. Omitting
`--include-embeddings` creates a vectorless backup; keyword search remains
available after restore, while hybrid retrieval requires reindexing. Portable
backups preserve logical IDs and source content but redact host-local paths.
Restore into a fresh database with a compatible binary, as described in
[the operating runbooks](operations.md). Detached manual notes retain their
source identity when generated records are pruned; metadata kept for provenance
is not repeatedly offered for deletion.

The [validation procedure](daily-workflow-validation.md) repeats the workflow
with isolated fixtures and records command counts, elapsed time, and automated
ID transfers. These are local automated measurements, not a human usability
study. Network/MCP access is separate work in
[#56](https://github.com/mateu/graphrag-notes/issues/56); native Linux and Intel
macOS acceptance remains in [#66](https://github.com/mateu/graphrag-notes/issues/66).
