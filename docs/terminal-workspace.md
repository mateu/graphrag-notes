# Daily terminal workspace

Run the latest source build with your usual configuration:

```sh
graphrag workspace
graphrag --config /path/to/config.toml workspace
```

`interactive` remains an alias. The published `v0.1.0-rc.1` binary predates this
workspace. macOS and Linux are supported.

The workspace keeps one database connection open until you quit. It starts
without contacting inference providers. Browse notes, inspect chats, search by
keyword, check source status, and review saved proposals with providers stopped.
Hybrid search and capture check the providers needed by that action; a failure
returns you to the prompt.

## Find, select, and reuse

```text
keyword Atlas
select 1
next
prev
copy
open
back
```

`keyword QUERY` searches notes and chats without inference. `search QUERY`
requests hybrid retrieval without silently falling back to another mode.
`recent [LIMIT]` lists visible notes. Limits are 1–200; the configured default
is capped at that bound in the workspace.

Each result shows its full ID and a revision. `select NUMBER` opens details;
`inspect`, `open`, and `copy` use the selected result without retyping its ID.
They also accept a displayed number or full ID. `next` and `prev` navigate the
same list. `back` first returns to the current list, then to a previous list.
Running another search or list clears the selection.

Numbers belong to the displayed snapshot. Every action rechecks the original
ID and revision. A changed, missing, or refreshed source cannot redirect an
old selection to another record; repeat the search to refresh it.

`open` uses the existing configured literal-argument source opener. A note or
chat without a local source file remains inspectable. `copy` prints selected
content for terminal reuse, with a JSON-quoted ID/revision citation on stderr;
it does not set the system clipboard. Terminal control characters are escaped
when displaying content in a terminal. This command runs inside a workspace
conversation with prompts and separators; use `augment --raw` for a standalone
prompt-context stream suitable for redirection.

## Capture and recover

```text
capture The Atlas launch checklist needs a rehearsal.
capture --editor
```

Inline capture takes the whole remaining line as literal note content.
`capture --editor` reuses the blocking editor and private recovery draft from
the ordinary capture command. Set `VISUAL` or `EDITOR` to a blocking editor
executable. For literal editor arguments, files, titles, tags, and structured
output, use the ordinary `graphrag capture` command.

Capture never consumes the workspace command stream as note input. It requires
persistent storage; `--memory workspace` supports ephemeral browsing but
refuses recoverable capture. Provider/editor failures or safe cancellation
retain the draft and print recovery guidance. Quit the workspace before
replaying that standalone command against the same database.

Ctrl-C clears unfinished terminal input or requests a safe stop during an
action. Read-only work can stop immediately. Capture stops before persistence
when possible; an atomic save already underway finishes and reports its saved
ID. A cancelled action does not cancel the next action. An external opener or
editor may need to finish before control returns.

## Sources and connections

```text
status
stats
proposals
select 1
review
```

`status [LIMIT]` shows source ingestion state, current/successful generations,
and the last error. `stats` shows corpus counts. Folder sync and import remain
standalone commands; quit first so those commands can own the database.

`proposals [all] [LIMIT]` lists pending proposals, or includes lifecycle
history. Select a proposal to see both endpoints and its evidence. `review`
asks you to choose `accept`, `reject`, `undo`, or `skip`, enter an optional
audit reason, and type `yes` to confirm a decision. `:cancel` skips from any
review stage; Ctrl-C clears unfinished terminal confirmation. Skip, declined
confirmation, and EOF write no decision. A changed proposal or endpoint must be
displayed again before a decision can apply. The workspace reuses the existing
governed policy and preserves its audit and undo behavior.

## Keyboard and storage ownership

On a terminal, Up/Down recall commands and Tab completes commands, displayed
IDs, and review choices. History is bounded and kept only in memory; it is
discarded on exit. `history` prints the current session history. Vertical
result/details layouts work in narrow terminals without a full-screen UI.

Piped commands use deterministic line input without terminal styling. Displayed
records escape control characters; `copy` preserves the original content in its
piped payload, including any control characters stored in that content:

```sh
printf 'recent\nselect 1\ninspect\nquit\n' | graphrag workspace
```

`quit` or EOF closes the connection. While the workspace owns persistent
storage, another CLI command targeting that database reports that it is in
use and asks you to exit the owner and retry. It cannot attach to the embedded
database. Future cross-computer access will use a single service-owned corpus;
see the [application boundary](application-boundary.md) and issue #56.

Validation evidence and limitations are in
[terminal workspace validation](terminal-workspace-validation.md).
